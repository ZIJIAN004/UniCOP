"""
HLR 训练数据加载: 从 profiled jsonl 构建分段训练样本.

每条样本拆成:
  - teacher: prompt + <think>完整 CoT</think> + solution    (完整显式推理)
  - student: prompt + 显式段 + latent 段 (k 步) + ... + solution 段
             latent 段没有 token id, 只存 k + teacher_align_pos + teacher_input_pos,
             训练时由 compute_hlr_loss 调 LatentReasoner 生成 hidden 注入 inputs_embeds.

Phase 1 限制: collate 强制 batch_size=1, 不做跨样本 padding.
"""

import json
import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

_POSTHOC_SYSTEM_MARKER = "\n\nYour output MUST start with <think>"
_POSTHOC_USER_MARKER = "\n\nTarget solution ("


def strip_posthoc(text: str, marker: str) -> str:
    idx = text.find(marker)
    return text[:idx] if idx != -1 else text


@dataclass
class HLRSegment:
    """单个段：explicit / latent / solution。"""
    type: str                                # "explicit" | "latent" | "solution"
    ids: torch.Tensor | None = None          # explicit/solution: token ids
    labels: torch.Tensor | None = None       # explicit/solution: labels (-100 屏蔽)
    k: int | None = None                     # latent: 要生成的 latent 数
    teacher_align_pos: int | None = None     # latent: 段末位对齐到 teacher 的绝对索引
    teacher_input_pos: int | None = None     # latent: LR input 对应 teacher 位置 (段起始前一个 token)


@dataclass
class HLRSample:
    teacher_input_ids: torch.Tensor
    teacher_labels: torch.Tensor
    prompt_ids: torch.Tensor                 # 单独存，方便主模型分段 forward
    segments: list[HLRSegment]


class HLRDataset(Dataset):
    """
    分段构造 HLR 训练样本。

    Teacher:
        prompt + <think>完整CoT</think> + solution + EOS
        labels 在整段 output 计 CE (与 Distill Stage 2 completion-only loss 等价)

    Student segments:
        [prompt_ids]   ── 不放进 segments，单独保存
        然后依次:
          explicit (CoT 段前部分)
          latent   (k_1)
          explicit (段间)
          latent   (k_2)
          ...
          solution (</think>\n + solution + EOS)
    """

    def __init__(self, data_path: str, tokenizer,
                 max_length: int = 8192,
                 latent_compression_ratio: int = 4,
                 filter_problems=None, filter_sizes=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.compression_ratio = latent_compression_ratio
        self.samples = []
        # 收集被成功 build 的 record 子集做覆盖率统计 (与 entropy_profile 端共用 schema)
        accepted_records = []

        n_invalid = 0          # 跳过 (json 解析失败 / output 空 / </think> 缺失)
        n_truncated = 0        # 因 teacher_ids > max_length 而 drop
        n_filtered = 0         # 因 filter_problems / filter_sizes 而 drop
        record_total_len_before_filter = 0  # 用于截断率分母
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    n_invalid += 1
                    continue

                if filter_problems and r.get("problem_type") not in filter_problems:
                    n_filtered += 1
                    continue
                if filter_sizes and r.get("n") not in filter_sizes:
                    n_filtered += 1
                    continue

                output = r.get("output", "")
                if not output or not output.strip():
                    n_invalid += 1
                    continue

                record_total_len_before_filter += 1
                sample, reason = self._build_sample_with_reason(r)
                if sample is not None:
                    self.samples.append(sample)
                    accepted_records.append(r)
                elif reason == "truncated":
                    n_truncated += 1
                else:
                    n_invalid += 1

        print(f"  HLRDataset 加载: {len(self.samples)} 接受 | "
              f"{n_truncated} 截断 (max_length={self.max_length}) | "
              f"{n_invalid} 无效 | {n_filtered} 被 filter")

        # ── 长度分布报送 (帮助调 max_length / Distill 数据生成端) ──
        if self.samples:
            import statistics
            lens = sorted(s.teacher_input_ids.size(0) for s in self.samples)
            n = len(lens)
            print(f"  Teacher token 长度分布 (接受样本 n={n}):")
            print(f"    min/max          : {lens[0]} / {lens[-1]}")
            print(f"    mean / median    : {sum(lens)/n:.0f} / {lens[n // 2]}")
            print(f"    p50/p75/p90/p95/p99: "
                  f"{lens[int(0.50*n)]}/{lens[int(0.75*n)]}/{lens[int(0.90*n)]}/"
                  f"{lens[int(0.95*n)]}/{lens[min(int(0.99*n), n-1)]}")
            if record_total_len_before_filter > 0:
                trunc_rate = n_truncated / record_total_len_before_filter
                print(f"  截断率: {n_truncated}/{record_total_len_before_filter} "
                      f"= {100*trunc_rate:.2f}%  (越高说明 max_length 越小, 浪费数据)")

        # ── token-level 压缩比例报送 ──
        try:
            from entropy_profile import summarize_latent_coverage, print_coverage_summary
            stats = summarize_latent_coverage(accepted_records, compression_ratio=self.compression_ratio)
            if stats["cot_total_tokens"] > 0:
                print_coverage_summary(stats, title="Latent 覆盖统计 (HLRDataset 训练侧)")
            else:
                print("  ⚠ 训练样本里没有 cot_token_count 字段 (旧版 profiled jsonl?), 跳过覆盖率报送")
        except ImportError:
            pass

    def _build_sample_with_reason(self, record):
        """包装 _build_sample, 返回 (sample, reason) 以区分截断 vs 其它无效."""
        # 简单实现: 先 tokenize 一次看长度, 再走完整 build
        # 如果未来想避免双重 tokenize, 可以重写 _build_sample 让它返回 reason
        try:
            sample = self._build_sample(record)
            if sample is not None:
                return sample, "ok"
            # _build_sample 返回 None 的原因可能是: 截断 / </think> 缺失 / teacher_align_pos OOB
            # 这里再做一次轻量截断探测 (复用 tokenizer 但只 encode 一次)
            output = record.get("output", "")
            if "</think>" not in output:
                return None, "no_think_close"
            # 估算长度判断是否截断
            tokenizer = self.tokenizer
            from data_utils import _POSTHOC_SYSTEM_MARKER, _POSTHOC_USER_MARKER, strip_posthoc
            sys_text = strip_posthoc(record["prompt"]["system"], _POSTHOC_SYSTEM_MARKER)
            usr_text = strip_posthoc(record["prompt"]["user"], _POSTHOC_USER_MARKER)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_text}, {"role": "user", "content": usr_text}],
                tokenize=False, add_generation_prompt=True,
            )
            if not prompt_text.rstrip().endswith("<think>"):
                prompt_text += "<think>\n"
            output_stripped = output.lstrip()
            if output_stripped.startswith("<think>"):
                output_stripped = output_stripped[len("<think>"):].lstrip("\n")
            est_ids = tokenizer.encode(prompt_text + output_stripped + tokenizer.eos_token,
                                       add_special_tokens=False)
            if len(est_ids) > self.max_length:
                return None, "truncated"
            return None, "invalid"
        except Exception:
            return None, "invalid"

    def _build_sample(self, record):
        """
        数据处理对齐 UniCOP-Distill/stage2_reasoning/train_sft_stage2.py 的 load_sft_dataset:
          - strip_posthoc 剥后验标记 (template 数据无标记时穿透)
          - chat_template + add_generation_prompt, fallback 补 <think>\\n
          - output 剥重复 <think>, 余下原样保留 (不再 hardcode </think>\\n)
          - teacher_text = prompt + output_stripped + eos (Distill 风原样)
          - teacher CE 整段 output 都算 (与 Distill Stage 2 completion-only loss 等价)
        """
        tokenizer = self.tokenizer

        orig_system = strip_posthoc(record["prompt"]["system"], _POSTHOC_SYSTEM_MARKER)
        orig_user = strip_posthoc(record["prompt"]["user"], _POSTHOC_USER_MARKER)
        output = record.get("output", "")
        if not output or not output.strip():
            return None

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": orig_system},
             {"role": "user", "content": orig_user}],
            tokenize=False, add_generation_prompt=True,
        )
        if not prompt_text.rstrip().endswith("<think>"):
            prompt_text += "<think>\n"

        # Output: 剥重复 <think>, 余下原样保留
        output_stripped = output.lstrip()
        if output_stripped.startswith("<think>"):
            output_stripped = output_stripped[len("<think>"):].lstrip("\n")

        think_close_idx = output_stripped.find("</think>")
        if think_close_idx == -1:
            return None

        cot_text = output_stripped[:think_close_idx]
        # post_think_text 含 </think> 自身 + raw 换行 + solution (原始格式保留, 不 lstrip 换行)
        post_think_text = output_stripped[think_close_idx:]
        if not post_think_text[len("</think>"):].strip():
            return None

        # ── Teacher 序列 (Distill 风原样拼接) ──
        teacher_text = prompt_text + output_stripped + tokenizer.eos_token
        teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)
        if len(teacher_ids) > self.max_length:
            return None

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Teacher CE 范围: 整段 output (cot + </think> + post_think + eos) 都算 CE.
        # 跟 Distill Stage 2 的 completion-only loss 语义一致 (保留显式推理能力)
        teacher_label_start = len(prompt_ids)
        teacher_labels = [-100] * teacher_label_start + teacher_ids[teacher_label_start:]

        # ── Student 分段构造 ──
        cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)
        # post_think_ids: </think> + 原始换行 + solution + eos (不再 hardcode "</think>\n")
        post_think_ids = tokenizer.encode(
            post_think_text + tokenizer.eos_token, add_special_tokens=False
        )
        # </think> 在 post_think_ids 开头占的 token 数 (后面 solution segment 用)
        think_close_search = tokenizer.encode("</think>", add_special_tokens=False)
        tc_len = len(think_close_search)

        teacher_cot_start = len(prompt_ids)
        latent_meta = record.get("latent_segments", [])

        segments: list[HLRSegment] = []
        cot_cursor = 0

        for seg in latent_meta:
            seg_start = max(seg["start"], cot_cursor)
            seg_end = seg["end"]
            if seg_end < seg_start:
                continue
            orig_len = seg_end - seg_start + 1
            k = max(1, math.ceil(orig_len / self.compression_ratio))

            # 段前显式
            if seg_start > cot_cursor:
                explicit_ids = cot_ids[cot_cursor:seg_start]
                segments.append(HLRSegment(
                    type="explicit",
                    ids=torch.tensor(explicit_ids, dtype=torch.long),
                    labels=torch.tensor(explicit_ids, dtype=torch.long),
                ))

            teacher_align_pos = teacher_cot_start + seg_end
            teacher_input_pos = max(0, teacher_cot_start + seg_start - 1)
            # teacher_input_pos 取段起始前一个 teacher 位置;
            # 若段从 cot 开头开始 (seg_start=0), 退到 prompt 末位 (= teacher_cot_start - 1)
            if teacher_align_pos >= len(teacher_ids):
                # teacher 序列在 max_length 处被截断, latent 段对齐 anchor 不可用
                # → align loss 不准, 整条样本丢弃 (这种情况在 max_length=8192 下很少触发)
                return None
            segments.append(HLRSegment(
                type="latent",
                k=k,
                teacher_align_pos=teacher_align_pos,
                teacher_input_pos=teacher_input_pos,
            ))

            cot_cursor = seg_end + 1

        # CoT 尾部显式
        if cot_cursor < len(cot_ids):
            remaining = cot_ids[cot_cursor:]
            segments.append(HLRSegment(
                type="explicit",
                ids=torch.tensor(remaining, dtype=torch.long),
                labels=torch.tensor(remaining, dtype=torch.long),
            ))

        # solution 段: </think> 自身不计 loss (是 cot 边界), 之后 (含原始换行 + solution + eos) 计 loss
        sol_labels_list = [-100] * tc_len + post_think_ids[tc_len:]
        segments.append(HLRSegment(
            type="solution",
            ids=torch.tensor(post_think_ids, dtype=torch.long),
            labels=torch.tensor(sol_labels_list, dtype=torch.long),
        ))

        # 粗略长度上限检查 (prompt + 所有段实际占位)
        total_len = len(prompt_ids) + sum(
            (s.ids.size(0) if s.ids is not None else (s.k or 0))
            for s in segments
        )
        if total_len > self.max_length:
            return None

        return HLRSample(
            teacher_input_ids=torch.tensor(teacher_ids, dtype=torch.long),
            teacher_labels=torch.tensor(teacher_labels, dtype=torch.long),
            prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
            segments=segments,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_hlr(batch: list[HLRSample], pad_token_id: int):
    """
    Phase 1: 强制 batch_size=1，不做跨样本 padding。
    返回字典里的 segments / prompt_ids 直接传给 model 的分段 forward。
    """
    assert len(batch) == 1, "HLR Phase 1 仅支持 per_device_batch_size=1"
    sample = batch[0]

    return {
        "teacher_input_ids": sample.teacher_input_ids.unsqueeze(0),
        "teacher_attention_mask": torch.ones_like(sample.teacher_input_ids).unsqueeze(0),
        "teacher_labels": sample.teacher_labels.unsqueeze(0),
        "prompt_ids": sample.prompt_ids,
        "segments": sample.segments,
    }
