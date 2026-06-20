"""stride>1 思维链 system 覆盖 (RL 训练 + RL eval 共用)。

prob.build_prompt 给的是 stride=1 的 _SYSTEM, 但 SFT 训练数据 (build_think_chains --stride N)
用的是 problems_prompt.get_system_prompt(pt, N) 的批量版 system。RL 训练/eval 若不替换, prompt
与 SFT 模型期望的不一致 (train/eval 失配)。这里用与 build_think_chains 同一函数 (单一真源) 覆盖
system, 保证逐字一致; user (实例描述) 不变。
"""
import os
import sys

_GET = None


def _get_system_prompt():
    global _GET
    if _GET is None:
        d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "UniCOP-Distill"))
        if d not in sys.path:
            sys.path.insert(0, d)
        from problems_prompt import get_system_prompt
        _GET = get_system_prompt
    return _GET


def apply_stride_system(prompt: list, problem_type: str, stride: int) -> list:
    """stride>1 时仅替换 system 为 SFT 训练同源的 stride 版; 否则原样返回。"""
    if not stride or stride <= 1:
        return prompt
    new_sys = _get_system_prompt()(problem_type, stride)
    return [{"role": "system", "content": new_sys} if m["role"] == "system" else m
            for m in prompt]
