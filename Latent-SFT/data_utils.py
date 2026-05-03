"""
数据加载：从 UniCOP-Distill 的 chains.jsonl 构建 CODI 训练数据。

每条样本需要拆分为：
  - teacher: prompt + <think>CoT</think> + solution  (完整显式推理)
  - student: prompt + <think> + [latent]*K + </think> + solution  (隐式推理)
  - align_token: solution 部分第一个 token 的位置 (对齐点)
"""

import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

_POSTHOC_SYSTEM_MARKER = "\n\nYour output MUST start with <think>"
_POSTHOC_USER_MARKER = "\n\nTarget solution ("

LATENT_TOKEN = "<latent>"

SPECIAL_TOKENS = [LATENT_TOKEN]


def strip_posthoc(text: str, marker: str) -> str:
    idx = text.find(marker)
    return text[:idx] if idx != -1 else text


def add_special_tokens(tokenizer):
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    print(f"  添加特殊 token: {LATENT_TOKEN}={latent_id}")
    return added, latent_id


@dataclass
class CODISample:
    teacher_input_ids: torch.Tensor
    teacher_labels: torch.Tensor
    student_input_ids: torch.Tensor
    student_labels: torch.Tensor
    latent_positions: list[int]
    teacher_align_pos: int
    student_align_pos: int


class CODIDataset(Dataset):
    """
    构建 CODI 训练对：teacher (显式 CoT) + student (latent tokens)。

    Teacher 序列:
      ...<|Assistant|><think>[CoT]</think>[solution]
                                           ^--- align position

    Student 序列:
      ...<|Assistant|><think>[<latent>]*K</think>[solution]
                                                  ^--- align position
    """

    def __init__(self, data_path: str, tokenizer, num_latent_tokens: int,
                 max_length: int = 8192,
                 filter_problems=None, filter_sizes=None):
        self.tokenizer = tokenizer
        self.num_latent_tokens = num_latent_tokens
        self.max_length = max_length
        self.samples = []

        latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)

        skipped = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                if filter_problems and r.get("problem_type") not in filter_problems:
                    continue
                if filter_sizes and r.get("n") not in filter_sizes:
                    continue

                output = r.get("output", "")
                if not output or not output.strip():
                    skipped += 1
                    continue

                sample = self._build_sample(r, latent_id)
                if sample is not None:
                    self.samples.append(sample)

        if skipped:
            print(f"  跳过 {skipped} 条无效记录")
        print(f"  成功加载 {len(self.samples)} 条 CODI 训练样本")

    def _build_sample(self, record, latent_id):
        tokenizer = self.tokenizer
        orig_system = strip_posthoc(record["prompt"]["system"], _POSTHOC_SYSTEM_MARKER)
        orig_user = strip_posthoc(record["prompt"]["user"], _POSTHOC_USER_MARKER)
        output = record["output"]

        # 渲染 prompt（到 assistant 标记为止）
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": orig_system},
             {"role": "user", "content": orig_user}],
            tokenize=False, add_generation_prompt=True,
        )
        # R1-Distill 的 add_generation_prompt 末尾带 <think>\n，兜底手动补
        if not prompt_text.rstrip().endswith("<think>"):
            prompt_text += "<think>\n"

        # 分离 CoT 和 solution
        output_stripped = output.lstrip()
        if output_stripped.startswith("<think>"):
            output_stripped = output_stripped[len("<think>"):].lstrip("\n")

        think_end = output_stripped.find("</think>")
        if think_end == -1:
            return None

        cot_text = output_stripped[:think_end]
        solution_text = output_stripped[think_end + len("</think>"):]
        solution_text = solution_text.lstrip("\n")

        if not solution_text.strip():
            return None

        # ── Teacher 序列 ──
        # ...<|Assistant|><think>\n + cot + </think>\n + solution + eos
        teacher_text = prompt_text + cot_text + "</think>\n" + solution_text + tokenizer.eos_token
        teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)

        if len(teacher_ids) > self.max_length:
            return None

        # 找 teacher 的对齐位置: </think> 子序列之后的第一个 solution token
        # R1-Distill tokenizer 中 </think> 被切分为多个 sub-token，需要做子序列匹配
        think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        tc_len = len(think_close_ids)
        teacher_align_pos = None
        for idx in range(len(teacher_ids) - tc_len, -1, -1):
            if teacher_ids[idx:idx + tc_len] == think_close_ids:
                teacher_align_pos = idx + tc_len
                break
        if teacher_align_pos is None or teacher_align_pos >= len(teacher_ids):
            return None

        # Teacher labels: 只在 solution 部分计算 loss
        teacher_labels = [-100] * teacher_align_pos + teacher_ids[teacher_align_pos:]

        # ── Student 序列 ──
        # ...<|Assistant|><think>\n + <latent>*K + </think>\n + solution + eos
        # prompt_text 已经以 <think>\n 结尾，直接复用
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        latent_block = [latent_id] * self.num_latent_tokens
        think_close_with_nl = tokenizer.encode("</think>\n", add_special_tokens=False)
        solution_with_eos = solution_text + tokenizer.eos_token
        solution_ids = tokenizer.encode(solution_with_eos, add_special_tokens=False)

        student_ids = prompt_ids + latent_block + think_close_with_nl + solution_ids

        if len(student_ids) > self.max_length:
            return None

        # Student 对齐位置: </think>\n 之后的第一个 solution token
        student_align_pos = len(prompt_ids) + len(latent_block) + len(think_close_with_nl)

        # Student labels: 只在 solution 部分计算 loss
        student_labels = [-100] * student_align_pos + solution_ids

        # Latent positions: <latent> tokens 在 student 序列中的绝对位置
        latent_start = len(prompt_ids)
        latent_positions = list(range(latent_start, latent_start + self.num_latent_tokens))

        return CODISample(
            teacher_input_ids=torch.tensor(teacher_ids, dtype=torch.long),
            teacher_labels=torch.tensor(teacher_labels, dtype=torch.long),
            student_input_ids=torch.tensor(student_ids, dtype=torch.long),
            student_labels=torch.tensor(student_labels, dtype=torch.long),
            latent_positions=latent_positions,
            teacher_align_pos=teacher_align_pos,
            student_align_pos=student_align_pos,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_codi(batch: list[CODISample], pad_token_id: int):
    """动态 padding，teacher 和 student 分别 pad 到各自 batch 内最长。"""

    def pad_tensors(tensors, pad_value):
        max_len = max(t.size(0) for t in tensors)
        padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
        masks = torch.zeros(len(tensors), max_len, dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, :t.size(0)] = t
            masks[i, :t.size(0)] = 1
        return padded, masks

    teacher_ids, teacher_masks = pad_tensors(
        [s.teacher_input_ids for s in batch], pad_token_id
    )
    teacher_labels, _ = pad_tensors(
        [s.teacher_labels for s in batch], -100
    )
    student_ids, student_masks = pad_tensors(
        [s.student_input_ids for s in batch], pad_token_id
    )
    student_labels, _ = pad_tensors(
        [s.student_labels for s in batch], -100
    )

    return {
        "teacher_input_ids": teacher_ids,
        "teacher_attention_mask": teacher_masks,
        "teacher_labels": teacher_labels,
        "student_input_ids": student_ids,
        "student_attention_mask": student_masks,
        "student_labels": student_labels,
        "latent_positions": [s.latent_positions for s in batch],
        "teacher_align_pos": [s.teacher_align_pos for s in batch],
        "student_align_pos": [s.student_align_pos for s in batch],
    }
