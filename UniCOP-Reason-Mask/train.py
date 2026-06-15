"""
主训练脚本：用 GRPO 微调 LLM 求解组合优化问题。

单卡运行：
    python train.py --problem tsptw --problem_size 10
    python train.py --problem tsp cvrp tsptw tspdl --num_gpus 1

多卡运行（需先安装 accelerate，并 accelerate config 完成环境配置）：
    accelerate launch --num_processes 4 train.py --problem tsptw --num_gpus 4 --zero_stage 3
    accelerate launch --num_processes 4 train.py --problem tsp cvrp tsptw tspdl --num_gpus 4 --zero_stage 3

注意：--num_gpus 必须与 --num_processes 保持一致，脚本不自动拉起多进程。
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.utils.checkpoint as torch_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
from peft import LoraConfig, get_peft_model

# ── 诊断: 开启 checkpoint debug 模式,出错时给出更详细的 tensor 信息 ────────
# 包括: tensor 来源代码位置、变量名、forward call 路径
# 性能损耗: 几乎为零 (只在元数据对比那一步多记录信息)
torch_checkpoint.set_checkpoint_debug_enabled(True)


# ── VLLMClient monkey-patch (两层): retry + logprobs 暴露 ─────────────
# Layer 1 (retry):
#   trl.extras.vllm_client.VLLMClient.generate 默认无 retry, 一旦 vLLM 进程
#   闪挂 (auto_train.sh 的 supervisor 正在重启) 立刻 RemoteDisconnected → exit 1.
#   包一层指数 backoff retry, vLLM 重启的 30-60 秒内 trainer 等待不死.
# Layer 2 (logprobs):
#   完全替换 VLLMClient.generate 实现, 始终发 return_logprobs=True 给服务端
#   (utils/vllm_serve_ngram.py 已注册新路由支持这个字段), 解析 response 拿
#   per-token sampled logprobs, 缓存到 self._last_logprobs 实例属性供训练端
#   读取算 IS ratio. 保持原方法签名返回 list[list[int]] (token_ids), 父类
#   GRPOTrainer._generate_and_score_completions 不受影响.
import time
import requests
import urllib3
from trl.extras.vllm_client import VLLMClient


def _patch_vllm_client_logprobs():
    """Layer 2: 替换 VLLMClient.generate 实现, 暴露 logprobs 到 self._last_logprobs.

    注意: 必须在 retry patch 之前装好, 让 retry 包装 layer 2 后的实现.
    """
    def _new_generate(
        self,
        prompts,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex=None,
    ):
        url = f"http://{self.host}:{self.server_port}/generate/"
        body = {
            "prompts": prompts,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "guided_decoding_regex": guided_decoding_regex,
            "return_logprobs": True,        # 始终拿 logprobs
            "return_mask_hits": True,       # 始终拿 mask_hits (server 未启用 mask 时返 None)
        }
        response = self.session.post(url, json=body)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        data = response.json()
        # 缓存 logprobs + mask_hits 到实例属性 (供训练端 _generate_and_score_completions 读)
        self._last_logprobs = data.get("logprobs")    # list[list[float]] 或 None
        self._last_mask_hits = data.get("mask_hits")  # list[list[bool]] 或 None
        return data["completion_ids"]

    VLLMClient.generate = _new_generate
    # 初始化默认值, 避免读未设置属性
    VLLMClient._last_logprobs = None
    VLLMClient._last_mask_hits = None
    print("✓ VLLMClient.generate replaced (始终发 return_logprobs=True + return_mask_hits=True, "
          "结果缓存到 self._last_logprobs / self._last_mask_hits)")


def _patch_vllm_client_retry(max_retries: int = 10, base_backoff: float = 3.0):
    """Layer 1: Monkey-patch VLLMClient.generate 和 update_named_param 加 retry.

    必须在 logprobs patch 之后调用, 让 retry 包装 logprobs-aware 的实现.
    """
    _orig_generate = VLLMClient.generate
    _orig_update_named_param = getattr(VLLMClient, "update_named_param", None)
    _exc_types = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        urllib3.exceptions.ProtocolError,
        ConnectionError,
        ConnectionResetError,
    )

    def _retry(orig_fn, fn_name):
        def wrapped(self, *args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return orig_fn(self, *args, **kwargs)
                except _exc_types as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    wait = base_backoff * (2 ** min(attempt, 4))  # 3, 6, 12, 24, 48, 48...
                    print(
                        f"⚠️ VLLMClient.{fn_name} attempt {attempt+1}/{max_retries+1} "
                        f"failed: {type(e).__name__}: {str(e)[:120]}; "
                        f"等 {wait}s 后重试 (vLLM supervisor 应该正在重启 vLLM)",
                        flush=True,
                    )
                    time.sleep(wait)
            print(
                f"❌ VLLMClient.{fn_name} 重试 {max_retries+1} 次后仍失败, 抛出",
                flush=True,
            )
            raise last_exc
        return wrapped

    VLLMClient.generate = _retry(_orig_generate, "generate")
    # ⚠️ 不要给 update_named_param 加 retry。它内部是 NCCL broadcast + barrier(collective):
    # retry = 训练端多做一次广播而 vLLM 端没有 → barrier 广播序号 off-by-one 永久错位死锁。
    # 实测 2026-05-26: 跑 4 step 后训练端等 /broadcast_from/0/1808、vLLM 等 .../1/1807, 双双 5min 超时崩。
    # 且 plain v5 无 vLLM supervisor, retry 它也救不了(没东西重启)。让它失败快崩 + resume_from_checkpoint。
    # (generate 是纯 HTTP、无持久 collective, retry 安全, 保留。)
    print(
        f"✓ VLLMClient.generate retry patched (update_named_param 不 retry: 避免 collective 错位死锁); "
        f"max_retries={max_retries}, base_backoff={base_backoff}s"
    )

# 顺序敏感: logprobs patch 先装 (替换 generate 实现), 再装 retry (包装 logprobs-aware
# 版本). retry 反过来不行 - 它会包装原 _orig_generate, 后续 logprobs patch 把 retry
# 包装的版本替换掉, 失去 retry.
_patch_vllm_client_logprobs()
_patch_vllm_client_retry()

from config import config

# ── A/B 实验 env 覆盖(不动 config.py 全局默认): 临时调 batch / grad_accum ──
# 例(测"不分片 ZeRO-2"提速): PER_DEVICE_BATCH=2 GRAD_ACCUM=24 ZERO_STAGE=2 + 4卡单NUMA。
# num_generations 不覆盖(保持 8 路对比)。⚠️ 整除约束: per_device_batch × num_gpus % num_generations == 0。
config.per_device_train_batch_size = int(os.environ.get("PER_DEVICE_BATCH", config.per_device_train_batch_size))
config.gradient_accumulation_steps = int(os.environ.get("GRAD_ACCUM", config.gradient_accumulation_steps))
# 消融实验: DISABLE_PRM=1 关闭 POMO PRM 过程奖励 (只留 A_feas + A_outcome), 且下面不加载 POMO。
if os.environ.get("DISABLE_PRM") is not None:
    config.disable_prm = os.environ.get("DISABLE_PRM") == "1"
# NUM_GEN: 覆盖 num_generations (默认 8)。单卡诊断用 4 → per_device_batch 可降到 4 避免 OOM 污染计时。
# ⚠️ 整除约束: per_device_batch × num_gpus % num_generations == 0 (run script 同步用 NUM_GEN 校验)。
if os.environ.get("NUM_GEN") is not None:
    config.num_generations = int(os.environ.get("NUM_GEN"))
# SAVE_STEPS: 覆盖 checkpoint 保存间隔 (默认 50)。某些实验要更密的存档 (如 noprm 用 20)。
if os.environ.get("SAVE_STEPS") is not None:
    config.save_steps = int(os.environ.get("SAVE_STEPS"))
# LR / EPOCHS: 覆盖学习率 / epoch (默认 config 值)。对照实验用 (如 v6 温和: LR=1e-6 EPOCHS=1)。
if os.environ.get("LR") is not None:
    config.learning_rate = float(os.environ.get("LR"))
if os.environ.get("EPOCHS") is not None:
    config.num_train_epochs = int(os.environ.get("EPOCHS"))
# ── v6 reward scheme 参数 env 覆盖 (reward_scheme=v6 时生效, 其余 scheme 忽略) ──
# 扫参用, 不改 config.py 默认值。PROC_ALPHA_V6 是 v6 主轴 (PRM 段注入权重);
# TRIM/S_MIN/S_MAX 是批级标准化的鲁棒性/数值守卫, 一般留默认。
if os.environ.get("PROC_ALPHA_V6") is not None:
    config.proc_alpha_v6 = float(os.environ.get("PROC_ALPHA_V6"))
if os.environ.get("TRIM_FRAC_V6") is not None:
    config.trim_frac_v6 = float(os.environ.get("TRIM_FRAC_V6"))
if os.environ.get("S_MIN_V6") is not None:
    config.s_min_v6 = float(os.environ.get("S_MIN_V6"))
if os.environ.get("S_MAX_V6") is not None:
    config.s_max_v6 = float(os.environ.get("S_MAX_V6"))
# ── A_feas 权重 env 覆盖 (v5/v6 共用; v6 经 _build_unified_advantages_v6 → _compute_a_out_v5
#    读 config.w_*_v5, grpo_prm_trainer.py:1990-1994)。扫参/对齐 FOARL 用, 不改 config.py 全局默认。
#    ⚠️ 改这些只改 A_feas 各分量相对主导地位; 若改了 A_feas 总量, 需同步核对
#       proc_alpha_v6 / A_outcome 标定 (它们按 A_feas≈5.5 量级设计)。
if os.environ.get("W_P_V5") is not None:
    config.w_p_v5 = float(os.environ.get("W_P_V5"))
if os.environ.get("W_COV_V5") is not None:
    config.w_cov_v5 = float(os.environ.get("W_COV_V5"))
if os.environ.get("W_CONS_V5") is not None:
    config.w_cons_v5 = float(os.environ.get("W_CONS_V5"))
if os.environ.get("W_F_V5") is not None:
    config.w_f_v5 = float(os.environ.get("W_F_V5"))

from data.generate import build_dataset, build_mixed_dataset
from problems import get_problem, SUPPORTED_PROBLEMS
from pomo_prm import POMOPRM
from grpo_prm_trainer import GRPOPRMTrainer
from terminal_reward import compute_terminal_reward


# ── DeepSpeed ZeRO 配置生成 ──────────────────────────────────────────────────

def make_deepspeed_config(zero_stage: int) -> dict | None:
    """
    根据 zero_stage 生成 DeepSpeed 配置字典，传给 GRPOConfig(deepspeed=...)。
    返回 None 表示不启用 DeepSpeed（单卡模式）。

    ZeRO-2：优化器状态 + 梯度分片，模型权重不拆分，通信开销低，适合 1.5B 多卡。
    ZeRO-3：完全分片（权重 + 梯度 + 优化器），每卡只保存 1/N 权重，7B 多卡必须用。
    """
    if zero_stage == 0:
        return None

    base = {
        "bf16":              {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size":               "auto",
        "gradient_accumulation_steps":    "auto",
        "steps_per_print":                50,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr":           "auto",
                "betas":        "auto",
                "eps":          "auto",
                "weight_decay": "auto",
            },
        },
    }

    if zero_stage == 2:
        base["zero_optimization"] = {
            "stage":                2,
            "overlap_comm":         True,
            "contiguous_gradients": True,
            "reduce_bucket_size":   5e8,
            "reduce_scatter":       True,
        }
        # ZeRO-2 只 offload 优化器状态 (参数不分片, 不 offload param)。
        # LoRA 可训练仅 ~132M, 优化器状态分片后每卡 ~0.26GB, 搬 CPU 给 ZeRO-2 压线腾这点显存。
        # 关键: 不像 ZeRO-3 param offload 每层搬权重 → 这里只在 optimizer.step 搬 132M 梯度/状态,
        # CPUAdam 在 132M 上很快, 速度代价可忽略。DS_OFFLOAD=0 可关 (要装 CUDA dev 头走 GPU FusedAdam)。
        if os.environ.get("DS_OFFLOAD", "1") != "0":
            base["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    elif zero_stage == 3:
        base["zero_optimization"] = {
            "stage":                          3,
            "overlap_comm":                   True,
            "contiguous_gradients":           True,
            "reduce_bucket_size":             "auto",
            "stage3_prefetch_bucket_size":    "auto",
            "stage3_param_persistence_threshold": 1_000_000,
            "stage3_max_live_parameters":     1e8,
            "stage3_max_reuse_distance":      1e9,
            "gather_16bit_weights_on_model_save": True,
        }
        # CPU offload 开关 (DS_OFFLOAD=0 关闭两个 offload, 默认开)。
        # 分片(ZeRO-3)是刚需(长序列激活大, 不分片必 OOM); offload 可去 → 关后基座仍分片在 GPU
        # (~1.4GB/卡), 不再每层从 CPU 经 PCIe 搬 → fwd+bwd 实测 ~50%。
        # ⚠️ 前置: DS_OFFLOAD=0 关 offload_optimizer 后 DeepSpeed 改用 FusedAdam(GPU), 首次会 JIT
        #   编译 multi_tensor_adam.cu, 需 env 有完整 CUDA toolkit(cuda_runtime.h 等 dev 头)。
        #   envs/unicop 缺头文件会编译失败 → 先把 CUDA dev 头补齐(conda install cuda-toolkit 匹配版本)。
        if os.environ.get("DS_OFFLOAD", "1") != "0":
            base["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            base["zero_optimization"]["offload_param"]     = {"device": "cpu", "pin_memory": True}

    return base


# ── 问题实例缓存 ─────────────────────────────────────────────────────────────
# get_problem() 每次调用都构造新对象，训练循环中频繁调用时产生无意义的重复开销。
# 问题类无内部状态（rng 由外部传入），同一类型复用同一实例完全安全。

_PROBLEM_CACHE: dict = {}

def _get_problem(problem_type: str):
    if problem_type not in _PROBLEM_CACHE:
        _PROBLEM_CACHE[problem_type] = get_problem(problem_type)
    return _PROBLEM_CACHE[problem_type]


# ── Placeholder reward（GRPOTrainer 强制要求至少一个 reward_func） ──────────
# 实际 reward 由 GRPOPRMTrainer 内部用 terminal_reward + POMO PRM 接管。
# 此函数仅为满足 trl 接口签名，返回值不被使用。

def _placeholder_reward_fn(completions, **kwargs):
    return [0.0] * len(completions)


# ── Token 长度 probe (R1 ↔ Qwen3 tokenizer 差异自适应) ──────────────────────
# 启动时跑 5 个真实样本, 测 prompt 和 completion 的实际 token 数, 推荐 max_*
# 值 (向上取 128 的倍数, 加 10% buffer). 对比当前 config 给 WARN.

def _ceil_128(x: int) -> int:
    """向上取整到 128 的倍数 (x*1.1 后取整, 留 10% buffer)."""
    if x <= 0:
        return 128
    target = int(x * 1.1)
    return ((target + 127) // 128) * 128


def _probe_token_lengths(trainer, tokenizer, problem_types, problem_size, num_samples=5):
    """跑 5 个样本, 测 prompt + completion 实际 token 数, 推荐 max_* 值."""
    import numpy as _np
    print("\n" + "=" * 70)
    print(f"  Token-length probe ({num_samples} samples, n={problem_size})")
    print(f"  目的: 测 prompt + completion 实际长度, 推荐 max_* 配置")
    print("=" * 70)

    rng = _np.random.default_rng(seed=2025)
    prompt_lens = []
    completion_lens = []

    vllm_client = getattr(trainer, "vllm_client", None)
    if vllm_client is None:
        print("  ⚠️ trainer.vllm_client 未就绪, 只测 prompt 长度")

    for i in range(num_samples):
        pt = problem_types[i % len(problem_types)]
        prob = _get_problem(pt)
        inst = prob.generate_instance(problem_size, rng)
        prompt_msgs = prob.build_prompt(inst)
        chat_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(
            chat_text, return_tensors="pt", add_special_tokens=False,
        ).input_ids
        p_len = prompt_ids.shape[1]
        prompt_lens.append(p_len)

        c_len_str = "(skipped)"
        if vllm_client is not None:
            try:
                comp_ids_list = vllm_client.generate(
                    prompts=[chat_text],
                    n=1,
                    temperature=float(getattr(trainer, "temperature", 1.0)),
                    top_p=float(getattr(trainer, "top_p", 1.0)),
                    top_k=-1 if getattr(trainer, "top_k", None) is None
                          else int(trainer.top_k),
                    max_tokens=config.max_completion_length,
                )
                c_len = len(comp_ids_list[0])
                completion_lens.append(c_len)
                c_len_str = f"{c_len}"
                if c_len >= config.max_completion_length - 5:
                    c_len_str += " (touched max_tokens cap!)"
            except Exception as _e:
                print(f"  样本 {i+1} vllm.generate 失败: {type(_e).__name__}: {_e}")

        print(f"  样本 {i+1} ({pt:>6} n={problem_size}): "
              f"prompt={p_len:>5}  completion={c_len_str}")

    if not prompt_lens:
        return

    max_p = max(prompt_lens)
    suggested_p = _ceil_128(max_p)

    print(f"\n  实测 prompt max = {max_p} → 推荐 max_prompt_length = {suggested_p}")
    print(f"    当前 config.max_prompt_length = {config.max_prompt_length} "
          f"{'✓' if config.max_prompt_length >= max_p else '❌ 太小, 训练会截 prompt!'}")

    if completion_lens:
        max_c = max(completion_lens)
        suggested_c = _ceil_128(max_c)
        suggested_mml = ((suggested_p + suggested_c + 256 + 127) // 128) * 128
        print(f"  实测 completion max = {max_c} → 推荐 max_completion_length = {suggested_c}")
        print(f"    当前 config.max_completion_length = {config.max_completion_length} "
              f"{'✓' if config.max_completion_length >= max_c else '❌ 太小'}")
        print(f"  推荐 VLLM_MAX_MODEL_LEN = {suggested_mml}")
        n_touched = sum(1 for c in completion_lens
                        if c >= config.max_completion_length - 5)
        if n_touched > 0:
            print(f"  ⚠️ {n_touched}/{len(completion_lens)} 样本 completion 触顶 "
                  f"max_completion_length={config.max_completion_length}, "
                  f"真实长度可能更大, 强烈建议调大 max_completion_length 重提.")

    print("=" * 70 + "\n")


# ── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem",      type=str, nargs="+", default=[config.problem_type],
                        choices=SUPPORTED_PROBLEMS,
                        help="一个或多个问题类型，多个时自动切换为混合训练模式")
    parser.add_argument("--problem_size", type=int, default=config.problem_size)
    parser.add_argument("--model",        type=str, default=config.model_name)
    parser.add_argument("--num_train",    type=int, default=config.num_train)
    parser.add_argument("--output_dir",   type=str, default=config.output_dir)
    parser.add_argument("--reward_mode",  type=str, default=config.reward_mode,
                        choices=["prm", "foarl"],
                        help="奖励模式：prm=三信号解耦+POMO PRM | foarl=FOARL 无 PRM")
    parser.add_argument("--reward_scheme", type=str,
                        default=getattr(config, "reward_scheme", "v5"),
                        choices=["v3", "v4", "v5", "v6"],
                        help="reward_mode=prm 时具体方案: "
                             "v3=hardgate+cascade | v4=simplified absolute PRM | "
                             "v5=v4+hardgate distance+cov/cons加权 (config 默认) | "
                             "v6=v5+PRM批级截尾标准化sigmoid")
    parser.add_argument("--no_lora",      action="store_true")
    parser.add_argument("--num_gpus",     type=int, default=config.num_gpus,
                        help="使用的 GPU 数量，需与 accelerate launch --num_processes 一致")
    parser.add_argument("--zero_stage",   type=int, default=config.zero_stage,
                        choices=[0, 2, 3],
                        help="DeepSpeed ZeRO stage：0=关闭 | 2=梯度分片(1.5B) | 3=完全分片(7B)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="开启梯度重计算节省显存，ZeRO-3 + 7B 时建议加上")
    parser.add_argument("--clip_epsilon_low", type=float,
                        default=config.clip_epsilon_low,
                        help="GRPO ratio clip 下界 ε_low (默认 0.20)")
    parser.add_argument("--clip_epsilon_high", type=float,
                        default=config.clip_epsilon_high,
                        help="GRPO ratio clip 上界 ε_high，> ε_low 启用 DAPO Clip-Higher (默认 0.28)")
    # POMO PRM (always enabled, vanilla reward 路径已删除)
    parser.add_argument("--pomo_ckpt_dir", type=str, default=config.pomo_ckpt_dir,
                        help="POMO checkpoint 根目录，子目录: {type}_n{size}/MODEL_BEST.pt")
    parser.add_argument("--pomo_baseline_dir", type=str, default=config.pomo_baseline_dir,
                        help="POMO-Baseline 项目根目录")
    parser.add_argument("--pipd_ckpt_dir", type=str, default=config.pipd_ckpt_dir,
                        help="PIP-D TSPTW checkpoint 根目录 (POMO+PIP/pretrained/TSPTW)")
    parser.add_argument("--pipd_dir", type=str, default=config.pipd_dir,
                        help="PIP-D 代码目录 (POMO+PIP),用于 sys.path 注入")
    parser.add_argument("--vllm_server_host", type=str, default="localhost",
                        help="vLLM server 主机，server 模式 rollout 加速用")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                        help="vLLM server 端口，需与 trl vllm-serve 启动端口一致")
    # ── Mask 超参 (跟 vLLM server 端 --mask_enabled / --mask_n 配对) ────
    parser.add_argument("--use_mask", action="store_true",
                        default=config.use_mask,
                        help="trainer 端 mask 开关；vLLM server 也必须用 "
                             "--mask_enabled --mask_n N 启动才会真生效")
    parser.add_argument("--mask_n", type=int, default=config.mask_n,
                        help="mask 的 customer 数 (0=跟 problem_size 同步)")
    parser.add_argument("--mask_debug", action="store_true",
                        default=config.mask_debug,
                        help="vLLM stderr 详细 mask 触发日志")
    args = parser.parse_args()

    problem_types = args.problem           # 始终为 list[str]
    is_mixed      = len(problem_types) > 1
    run_tag       = "mixed" if is_mixed else problem_types[0]

    config.problem_type  = problem_types[0]   # 保持向后兼容（evaluate 等处读单值）
    config.problem_size  = args.problem_size
    config.model_name    = args.model
    config.num_train     = args.num_train
    config.output_dir    = os.path.join(args.output_dir, f"{run_tag}_n{args.problem_size}")
    if args.no_lora:
        config.use_lora = False
    config.num_gpus               = args.num_gpus
    config.zero_stage             = args.zero_stage
    config.gradient_checkpointing = args.gradient_checkpointing or config.gradient_checkpointing
    config.clip_epsilon_low       = args.clip_epsilon_low
    config.clip_epsilon_high      = args.clip_epsilon_high
    config.reward_mode            = args.reward_mode
    config.reward_scheme          = args.reward_scheme
    config.pomo_ckpt_dir          = args.pomo_ckpt_dir
    config.pomo_baseline_dir      = args.pomo_baseline_dir
    config.pipd_ckpt_dir          = args.pipd_ckpt_dir
    config.pipd_dir               = args.pipd_dir
    config.use_mask               = args.use_mask
    config.mask_n                 = args.mask_n if args.mask_n > 0 else args.problem_size
    config.mask_debug             = args.mask_debug

    # ── 早期检查：PRM 模式下所有问题类型必须在 POMO PRM 支持列表内 ──
    if config.reward_mode == "prm":
        unsupported = [pt for pt in problem_types if pt not in POMOPRM.SUPPORTED]
        if unsupported:
            raise ValueError(
                f"以下问题类型不在 POMO PRM 支持列表 {sorted(POMOPRM.SUPPORTED)}: "
                f"{unsupported}。请使用 --reward_mode foarl 或仅使用 POMO 支持的类型。"
            )

    print(f"奖励模式:  {config.reward_mode}"
          f"{'（三信号解耦 + POMO PRM）' if config.reward_mode == 'prm' else '（FOARL 无 PRM）'}")
    print(f"问题类型:  {problem_types}{'（混合模式）' if is_mixed else ''}")
    print(f"问题规模:  n={config.problem_size}")
    print(f"模型:      {config.model_name}")
    print(f"训练样本:  {config.num_train}")
    print(f"输出路径:  {config.output_dir}")
    print(f"GPU 数量:  {config.num_gpus}  ZeRO stage: {config.zero_stage}"
          f"  梯度重计算: {config.gradient_checkpointing}")
    _ds_offload_on = (config.zero_stage == 3 and os.environ.get("DS_OFFLOAD", "1") != "0")
    print(f"CPU offload:  {'ON (param+optimizer→CPU, 慢)' if _ds_offload_on else 'OFF (全留 GPU)'}"
          f"   [DS_OFFLOAD={os.environ.get('DS_OFFLOAD', '1')}, 优化器={'CPUAdam' if _ds_offload_on else 'FusedAdam'}]")
    _clip_mode = ("asymmetric (Clip-Higher)"
                  if config.clip_epsilon_high > config.clip_epsilon_low
                  else "symmetric")
    print(f"Ratio clip:  ε_low={config.clip_epsilon_low}, "
          f"ε_high={config.clip_epsilon_high}  [{_clip_mode}]")
    print(f"Reward scheme: {getattr(config, 'reward_scheme', 'v3')}"
          f"  |  use_mask: {config.use_mask}"
          f"{f' (n={config.mask_n}, debug={config.mask_debug})' if config.use_mask else ''}")
    if config.use_mask and config.problem_type != "cvrp":
        print(f"⚠️ use_mask=True 但 problem_type={config.problem_type} 不是 cvrp; "
              f"mask 仅对 cvrp 实现, 其他问题类型不会被 vLLM mask 拦截.")

    # ── 加载模型 ────────────────────────────────────────────────────────
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    # ⚠️ pad_token 处理 (对齐 SFT 的 safe 逻辑):
    # GRPO 下 completion_mask 基于 attention_mask 识别,SFT 那种"EOS 被 mask
    # 不学"的致命 bug 在 GRPO 里 不直接发生。但仍然避免 pad=eos,
    # 保证 attention_mask/padding 逻辑清晰,和 SFT 阶段一致。
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        _pad_set = False
        for cand in ["<｜▁pad▁｜>", "<|▁pad▁|>", "<|PAD_TOKEN|>"]:
            tid = tokenizer.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id:
                tokenizer.pad_token = cand
                _pad_set = True
                break
        if not _pad_set:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    # ⚠️ 不要用 tokenizer.model_max_length 限制长度。
    # 原配置 `tokenizer.model_max_length = config.max_prompt_length (=768)` 会触发
    # "Token indices sequence length is longer than the specified maximum sequence length"
    # 警告 (超过 768 就 warn),但实际 prompt+completion 总长常到 5000 token。
    # 正确做法: 保留 tokenizer 从 model config 读到的上限 (Qwen2.5 是 131K),
    # prompt 长度上限由 GRPOConfig(max_prompt_length=...) 独立控制,见下面 grpo_config。

    # attn 实现: 装了 flash-attn 就用 flash_attention_2(zhuoyi/zhihan 两台已装),
    # 否则退回 sdpa(仍 O(S), 不会崩)。flash_attention_2 自身就是 O(S) 专用 kernel,
    # 上面 _sdpa_no_math 守卫只在退回 sdpa 时起作用(flash 路径不走 F.sdpa, 守卫自动空转)。
    try:
        import flash_attn as _fa  # noqa: F401
        _attn_impl = "flash_attention_2"
    except Exception:
        _attn_impl = "sdpa"
    print(f"attn 实现:  {_attn_impl}  "
          f"({'flash-attn 已装' if _attn_impl == 'flash_attention_2' else 'flash-attn 未装 → 退回 sdpa'})")

    # ── (可选) Liger Kernel 提速: 只融合 RMSNorm/RoPE/(可选)SwiGLU ───────────────
    # 这些 Triton kernel 在模型 forward 层加速 (减少 kernel 启动 + HBM 往返, ~10-20% 吞吐),
    # 与自定义 GRPO _compute_loss 完全解耦。必须关 cross_entropy / fused_linear_cross_entropy:
    # 我们手动算 completion logits 做 per-token logprob, 不走模型内置 loss 路径。
    # 默认关闭, 对现有训练零影响; 启用: export USE_LIGER=1
    #   SwiGLU 与 LoRA(gate/up/down_proj) 有交互, 先只开 rms_norm+rope 验证一版,
    #   稳定后再 export LIGER_SWIGLU=1 叠加。patch 必须在 from_pretrained 之前生效。
    if os.environ.get("USE_LIGER", "0") == "1":
        try:
            from transformers import AutoConfig
            _mt = AutoConfig.from_pretrained(
                config.model_name, trust_remote_code=True).model_type
            _use_swiglu = os.environ.get("LIGER_SWIGLU", "0") == "1"
            _liger_kwargs = dict(rms_norm=True, rope=True, swiglu=_use_swiglu,
                                 cross_entropy=False, fused_linear_cross_entropy=False)
            if _mt == "qwen3":
                from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as _apply_liger
            elif _mt == "qwen2":
                from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as _apply_liger
            else:
                _apply_liger = None
                print(f"⚠️ Liger: 未识别 model_type={_mt}, 跳过 (仅适配 qwen2/qwen3)")
            if _apply_liger is not None:
                _apply_liger(**_liger_kwargs)
                print(f"✓ Liger Kernel 已启用 (model_type={_mt}, swiglu={_use_swiglu}, "
                      f"cross_entropy/FLCE=关)")
        except ImportError:
            print("⚠️ USE_LIGER=1 但未装 liger-kernel, 跳过 (pip install liger-kernel)")
        except Exception as _e:
            print(f"⚠️ Liger 启用失败, 跳过: {type(_e).__name__}: {_e}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=_attn_impl,
    )

    # 如果前面 add_special_tokens 新加了 pad token (vocab size +1),
    # 必须在 LoRA wrap 之前 resize embedding,否则 pad_token_id 会超出 embedding 范围
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        print(f"  resize_token_embeddings {model.get_input_embeddings().num_embeddings} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if config.use_lora:
        print(f"启用 LoRA (rank={config.lora_rank})")
        lora_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # gradient_checkpointing + LoRA + ZeRO-3 三件套必需:
        # base 是 frozen 的, embedding 输出默认无 grad, checkpoint 在 backward
        # 时算不出反传路径,触发 "Recomputed values shape [0]" 错。
        # 这一行强制让 input embeddings 的输出 requires_grad=True,
        # 让 LoRA 反向传播能正确回到 input。
        if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            print("✓ enable_input_require_grads() 已开启 (LoRA + gc 必需)")

        # ── 诊断: 显式调 gradient_checkpointing_enable,确保 use_reentrant 设置落实 ─
        # TRL 1.1 内部也会调一次,这里我们先调,后面 trainer 看到已经开了就不再覆盖
        # 关键: 顺序必须是先 gc_enable,再 input_require_grads
        if config.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": True}
                )
                print("✓ gradient_checkpointing_enable() with use_reentrant=True")
                # 再 enable 一次 input_require_grads 以防顺序不对
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            except Exception as e:
                print(f"⚠️ gradient_checkpointing_enable failed: {e}")

        # ── 诊断: 打印前 5 个 frozen base param 的状态 ─────────────────────────
        print("\n=== 模型加载后的 frozen param 抽样 (DeepSpeed wrap 之前) ===")
        frozen_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad and frozen_count < 5:
                print(f"  [frozen] {name}: shape={tuple(param.shape)}, "
                      f"numel={param.numel()}, dtype={param.dtype}")
                frozen_count += 1
        print(f"  (注: ZeRO-3 wrap 后 < stage3_param_persistence_threshold "
              f"的小 param 应保持完整不分片)\n")

    # ── 数据集 ──────────────────────────────────────────────────────────
    print("\n生成训练数据...")
    if is_mixed:
        num_each = config.num_train // len(problem_types)
        train_dataset = build_mixed_dataset(
            problem_types=problem_types,
            num_samples_each=num_each,
            seed=config.data_seed,
            n=config.problem_size,
        )
    else:
        train_dataset = build_dataset(
            problem_type=problem_types[0],
            num_samples=config.num_train,
            seed=config.data_seed,
            n=config.problem_size,
        )
    print(f"训练集大小: {len(train_dataset)}")

    # ── GRPO 配置 ────────────────────────────────────────────────────────
    ds_config = make_deepspeed_config(config.zero_stage)

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        # GRPO/RL 标准用 constant_with_warmup, 不用 trl 默认的 linear (warmup → linear decay → 0).
        # linear decay 是 SFT 习惯, RL 不需要后期 LR 减小 (long-term reward optimization).
        # DeepSeek-R1 / DAPO / Verl 主流 GRPO 都用 constant. 实测 step 250 linear decay
        # 已经减半 LR, 250 step 后 update 速度 -50%, 拖慢长 run.
        lr_scheduler_type="constant_with_warmup",
        beta=config.kl_coef,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="no",
        save_strategy="steps",
        report_to="wandb" if config.use_wandb else "none",
        bf16=True,
        remove_unused_columns=False,
        deepspeed=ds_config,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        epsilon=config.clip_epsilon_low,
        epsilon_high=config.clip_epsilon_high,
        max_prompt_length=config.max_prompt_length,
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        # ── 采样参数 (paths.sh 按 BASE_MODEL_TYPE 注入 env, config 透传) ─────
        # 默认 GRPOConfig.temperature=1.0 是 R1-Distill 路径, Qwen3-Thinking 必须
        # 改 0.6 (官方推荐). 同时 trainer.{temperature,top_p,top_k} 用于
        # resample_infeasible 调 vLLM 时复用.
        temperature=config.gen_temperature,
        top_p=config.gen_top_p,
        top_k=config.gen_top_k if config.gen_top_k > 0 else None,
    )
    print(f"采样参数:  T={config.gen_temperature}  top_p={config.gen_top_p}  "
          f"top_k={config.gen_top_k} (BASE_MODEL_TYPE="
          f"{os.environ.get('BASE_MODEL_TYPE', '(unset)')})")

    # ── 初始化奖励模块 + 训练 ──────────────────────────────────────────
    # disable_prm=True (消融): 不加载 POMO, pomo_prm=None。trainer 三处 PRM 注入都
    # 由 `not config.disable_prm and self.pomo_prm is not None` 守卫, 自然跳过 →
    # 只剩 A_feas + A_outcome, 且不依赖 POMO checkpoint、省 POMO 那块显存。
    if config.reward_mode == "prm" and not config.disable_prm:
        pomo_prm = POMOPRM(
            pomo_ckpt_dir=config.pomo_ckpt_dir,
            pomo_baseline_dir=config.pomo_baseline_dir,
            device=config.pomo_device,
            pipd_ckpt_dir=config.pipd_ckpt_dir or None,
            pipd_dir=config.pipd_dir or None,
        )
        pomo_prm.check_checkpoints(problem_types, [config.problem_size])
    else:
        pomo_prm = None
        if config.disable_prm:
            print("⚠️ DISABLE_PRM: 消融模式, 不加载 POMO PRM, 只用 A_feas + A_outcome")

    trainer = GRPOPRMTrainer(
        pomo_prm=pomo_prm,
        problem_types=problem_types,
        model=model,
        reward_funcs=[_placeholder_reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    if pomo_prm is not None:
        pomo_prm.device = trainer.accelerator.device
        print(f"POMO PRM device: {pomo_prm.device}")

    print("\n开始 GRPO 训练...")

    # ── 诊断: 在 trainer.train() 之前打印 DeepSpeed 实际生效的 ZeRO 配置 ────
    # 让 trainer 先做完 DeepSpeed init, 但不真的开始训练
    # 这里通过 _wrap_model 触发 (transformers 内部入口)
    try:
        if hasattr(trainer, "deepspeed_plugin") and trainer.deepspeed_plugin is not None:
            print("\n=== DeepSpeed 实际生效配置 (zero_optimization 部分) ===")
            ds_cfg = trainer.deepspeed_plugin.deepspeed_config or {}
            zo = ds_cfg.get("zero_optimization", {})
            for k in ("stage", "stage3_param_persistence_threshold",
                      "stage3_max_live_parameters", "stage3_max_reuse_distance",
                      "stage3_prefetch_bucket_size", "reduce_bucket_size"):
                print(f"  {k}: {zo.get(k, '(not set)')}")
            print()
    except Exception as e:
        print(f"⚠️ 读取 DeepSpeed 配置失败: {e}")

    # ── 启动总览: 主要 step 数, 便于跟实测对照 ─────────────────────────
    # effective_batch_in_completions = per_device * num_processes * grad_accum
    # effective_batch_in_prompts     = effective_batch_in_completions / num_generations
    # total_train_steps              = num_train * num_epochs / effective_batch_in_prompts
    _eff_comp = (config.per_device_train_batch_size
                 * max(1, config.num_gpus)
                 * config.gradient_accumulation_steps)
    _eff_prompt = max(1, _eff_comp // config.num_generations)
    _total_steps = (config.num_train * config.num_train_epochs) // _eff_prompt
    print(
        f"\n=== 训练 step 总览 ===\n"
        f"  per_device_batch={config.per_device_train_batch_size}  num_gpus={config.num_gpus}  "
        f"grad_accum={config.gradient_accumulation_steps}  num_gen={config.num_generations}\n"
        f"  effective_batch (completions) = {_eff_comp}\n"
        f"  effective_batch (prompts)     = {_eff_prompt}\n"
        f"  total_train_steps             = {_total_steps}  "
        f"(num_train={config.num_train} × epochs={config.num_train_epochs} / {_eff_prompt})\n"
        f"  save_steps={config.save_steps}  logging_steps={config.logging_steps}  "
        f"resample_start_step={getattr(config, 'resample_start_step', 100)}\n"
    )

    # ── Token-length probe: 5 个真实样本, 推荐 max_prompt / max_completion ─
    # 防 Qwen3 ↔ R1 切换时 tokenizer 差异导致截断 (R1 估算的 max_* 在 Qwen3
    # 可能不够). 只在 main process 跑, 用 vLLM client 跑 5 个 completion 看实际长度.
    if trainer.accelerator.is_main_process:
        try:
            _probe_token_lengths(trainer, tokenizer, problem_types, args.problem_size)
        except Exception as _e:
            print(f"⚠️ token-length probe 失败 (不阻塞训练): {type(_e).__name__}: {_e}")

    # ── resume_from_checkpoint: 自动从最新 checkpoint 恢复 ──────────────
    # 配合 auto_train.sh 的 vLLM-disconnect 重试逻辑: vLLM 死亡后整个 job
    # 重启时, trainer.train(resume_from_checkpoint=True) 让 transformers
    # Trainer 自动找 output_dir 下最新 checkpoint-{step}, 加载 model/optimizer
    # /scheduler/rng 状态, 接着训.
    # - 第一次启动 (output_dir 下没 checkpoint): 自动从 step 0 开始, 无副作用
    # - 重启时 (有 checkpoint): 从最新 step 恢复
    _has_ckpt = os.path.isdir(config.output_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(config.output_dir)
    )
    if _has_ckpt:
        print(f"检测到 {config.output_dir} 下有 checkpoint, 从最新一份恢复训练...")
    trainer.train(resume_from_checkpoint=_has_ckpt)

    # ── 保存 ─────────────────────────────────────────────────────────────
    # trainer.save_model 内部已有 rank 守卫（ZeRO-3 gather + 主 rank 写）；
    # tokenizer.save_pretrained 没有，多 rank 同时写会竞争，必须显式守卫。
    save_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(save_path)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(save_path)
    trainer.accelerator.wait_for_everyone()
    print(f"\n模型已保存到: {save_path}")

    # _save_examples 已禁用（耗时 + 占显存）；推理样例由 evaluate.py 统一处理


def _save_examples(model, tokenizer, problem_types, n, save_dir, num_examples=3):
    """
    对每种问题类型各生成 num_examples 个推理样例并保存到 save_dir/examples.json。
    每条样例记录：
        - problem_type, instance_id
        - prompt_tokens:      prompt 的 token 数（与 max_prompt_length 比较判断是否截断）
        - completion_tokens:  生成的 token 数（与 max_completion_length 比较判断是否截断）
        - truncated:          completion 是否触达 max_completion_length 上限（疑似截断）
        - prompt_text:        完整 prompt 文本
        - completion_text:    模型生成的完整文本（含 <think> 链）
        - terminal_reward:    terminal reward 值（4 维加权和，∈ [0, 4]）
        - is_feasible:        严格可行性判断
    """
    print("\n生成推理样例...")
    model.eval()
    rng = np.random.default_rng(seed=2025)
    examples = []

    for pt in problem_types:
        prob = _get_problem(pt)
        for i in range(num_examples):
            instance   = prob.generate_instance(n, rng)
            prompt     = prob.build_prompt(instance)
            chat_text  = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            inputs     = tokenizer(chat_text, return_tensors="pt").to(model.device)
            prompt_tokens = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_completion_length,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            completion_ids    = outputs[0][prompt_tokens:]
            completion_tokens = len(completion_ids)
            # skip_special_tokens=False: 保 Qwen3-Thinking 的 </think> (special token)
            completion_text   = tokenizer.decode(completion_ids, skip_special_tokens=False)
            for _tok in ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>",
                         "<|begin_of_text|>", "<|eot_id|>"):
                completion_text = completion_text.replace(_tok, "")
            truncated         = (completion_tokens >= config.max_completion_length)

            instance_for_eval = prob.from_json(prob.to_json(instance))
            is_feasible = prob.is_feasible(completion_text, instance_for_eval)
            tour_dist   = prob.get_tour_distance(completion_text, instance_for_eval)
            term_reward = compute_terminal_reward(completion_text, instance_for_eval, pt)

            examples.append({
                "problem_type":      pt,
                "instance_id":       i,
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "truncated":         truncated,
                "is_feasible":       is_feasible,
                "terminal_reward":   round(term_reward, 4),
                "tour_distance":     round(tour_dist, 4) if tour_dist is not None else None,
                "prompt_text":       chat_text,
                "completion_text":   completion_text,
            })

            status = "✓可行" if is_feasible else "✗不可行"
            trunc  = " ⚠️截断" if truncated else ""
            print(f"  [{pt}] 样例{i+1}: terminal={term_reward:.3f} {status}"
                  f"  prompt={prompt_tokens}tok  completion={completion_tokens}tok{trunc}")

    out_path = os.path.join(save_dir, "examples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"样例已保存到: {out_path}")


if __name__ == "__main__":
    main()
