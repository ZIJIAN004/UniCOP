"""
数据加载：从 profiled jsonl 构建 CODI 混合训练数据。

每条样本拆分为：
  - teacher: prompt + <think>CoT</think> + solution  (完整显式推理)
  - student: prompt + <think>[混合 latent/explicit CoT]</think> + solution
  - align_pairs: 每个 latent→explicit 边界 + solution 边界的 (teacher, student) 位置对
"""

import json
import math
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
    align_pairs: list[tuple[int, int]]


class CODIDataset(Dataset):
    """
    从 profiled jsonl 构建 CODI 混合训练对。

    Teacher: ...<|Assistant|><think>[完整 CoT]</think>[solution]
    Student: ...<|Assistant|><think>[混合 latent/explicit]</think>[solution]

    对齐点: 每个 latent 段出口 + solution 入口
    """

    def __init__(self, data_path: str, tokenizer,
                 max_length: int = 8192,
                 latent_compression_ratio: int = 4,
                 filter_problems=None, filter_sizes=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.compression_ratio = latent_compression_ratio
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

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": orig_system},
             {"role": "user", "content": orig_user}],
            tokenize=False, add_generation_prompt=True,
        )
        if not prompt_text.rstrip().endswith("<think>"):
            prompt_text += "<think>\n"

        output_stripped = output.lstrip()
        if output_stripped.startswith("<think>"):
            output_stripped = output_stripped[len("<think>"):].lstrip("\n")

        think_end = output_stripped.find("</think>")
        if think_end == -1:
            return None

        cot_text = output_stripped[:think_end]
        solution_text = output_stripped[think_end + len("</think>"):].lstrip("\n")
        if not solution_text.strip():
            return None

        # ── Teacher 序列 ──
        teacher_text = (
            prompt_text + cot_text + "</think>\n"
            + solution_text + tokenizer.eos_token
        )
        teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)

        if len(teacher_ids) > self.max_length:
            return None

        # Teacher: 找 </think> 后的 \n 位置 (solution 对齐基准)
        think_close_search = tokenizer.encode("</think>", add_special_tokens=False)
        tc_len = len(think_close_search)
        teacher_solution_align = None
        for idx in range(len(teacher_ids) - tc_len, -1, -1):
            if teacher_ids[idx : idx + tc_len] == think_close_search:
                teacher_solution_align = idx + tc_len
                break
        if teacher_solution_align is None or teacher_solution_align >= len(teacher_ids):
            return None

        # Teacher labels: solution 部分
        teacher_label_start = teacher_solution_align + 1
        teacher_labels = [-100] * teacher_label_start + teacher_ids[teacher_label_start:]

        # ── Student 序列：混合 latent/explicit ──
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)
        think_close_nl = tokenizer.encode("</think>\n", add_special_tokens=False)
        solution_ids = tokenizer.encode(
            solution_text + tokenizer.eos_token, add_special_tokens=False
        )

        teacher_cot_start = len(prompt_ids)
        latent_segments = record.get("latent_segments", [])

        student_ids = list(prompt_ids)
        student_labels = [-100] * len(prompt_ids)
        latent_positions = []
        align_pairs = []

        cot_cursor = 0
        for seg in latent_segments:
            seg_start = max(seg["start"], cot_cursor)
            seg_end = seg["end"]
            orig_len = seg_end - seg_start + 1
            num_latent = max(1, math.ceil(orig_len / self.compression_ratio))

            # 段前的 explicit tokens
            if seg_start > cot_cursor:
                explicit = cot_ids[cot_cursor:seg_start]
                student_ids.extend(explicit)
                student_labels.extend(explicit)

            # Latent tokens
            latent_start = len(student_ids)
            student_ids.extend([latent_id] * num_latent)
            student_labels.extend([-100] * num_latent)
            latent_positions.extend(range(latent_start, latent_start + num_latent))

            # 对齐: student 最后一个 latent ↔ teacher CoT seg_end
            teacher_pos = teacher_cot_start + seg_end
            student_pos = latent_start + num_latent - 1
            if teacher_pos < len(teacher_ids):
                align_pairs.append((teacher_pos, student_pos))

            cot_cursor = seg_end + 1

        # 最后一段 latent 之后的 explicit CoT
        if cot_cursor < len(cot_ids):
            remaining = cot_ids[cot_cursor:]
            student_ids.extend(remaining)
            student_labels.extend(remaining)

        # </think>\n (不计 loss)
        student_ids.extend(think_close_nl)
        student_labels.extend([-100] * len(think_close_nl))

        # Solution 边界对齐
        student_solution_align = len(student_ids) - 1
        align_pairs.append((teacher_solution_align, student_solution_align))

        # Solution + EOS (计 loss)
        student_ids.extend(solution_ids)
        student_labels.extend(solution_ids)

        if len(student_ids) > self.max_length:
            return None

        return CODISample(
            teacher_input_ids=torch.tensor(teacher_ids, dtype=torch.long),
            teacher_labels=torch.tensor(teacher_labels, dtype=torch.long),
            student_input_ids=torch.tensor(student_ids, dtype=torch.long),
            student_labels=torch.tensor(student_labels, dtype=torch.long),
            latent_positions=latent_positions,
            align_pairs=align_pairs,
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
            padded[i, : t.size(0)] = t
            masks[i, : t.size(0)] = 1
        return padded, masks

    teacher_ids, teacher_masks = pad_tensors(
        [s.teacher_input_ids for s in batch], pad_token_id
    )
    teacher_labels, _ = pad_tensors([s.teacher_labels for s in batch], -100)
    student_ids, student_masks = pad_tensors(
        [s.student_input_ids for s in batch], pad_token_id
    )
    student_labels, _ = pad_tensors([s.student_labels for s in batch], -100)

    return {
        "teacher_input_ids": teacher_ids,
        "teacher_attention_mask": teacher_masks,
        "teacher_labels": teacher_labels,
        "student_input_ids": student_ids,
        "student_attention_mask": student_masks,
        "student_labels": student_labels,
        "latent_positions": [s.latent_positions for s in batch],
        "align_pairs": [s.align_pairs for s in batch],
    }
