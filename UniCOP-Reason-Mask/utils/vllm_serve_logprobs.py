"""
Wrapper: 替换 /generate/ 路由暴露 sampled-token logprobs + 可选 CVRP mask,
然后启动 trl vllm-serve.

两个 patch:
  1. /generate/ logprobs (始终启用):
     - 扩展 GenerateRequest 加 return_logprobs / return_mask_hits 字段
     - 扩展 GenerateResponse 加 logprobs / mask_hits 字段
     - sampling_params 用 logprobs=1, 从 output 取 sampled token logprob
     - 用于 GRPO Importance Sampling 校正 (vLLM vs train kernel 差异)

  2. CVRP mask processor (可选, --mask_enabled 控制):
     - 加载 CVRPMaskProcessor (utils/vllm_cvrp_mask_processor.py)
     - 通过 SamplingParams 实例字段赋值注入 (msgspec.Struct C-level __init__
       不能 monkey-patch, 调研结论)
     - 强制不重复访问 customer + 走完所有 n 个客户

用法:
    # 不启用 mask (跟 hardgate run 相同行为)
    python utils/vllm_serve_logprobs.py --model /path/to/model --port 8000

    # 启用 mask (新 run)
    python utils/vllm_serve_logprobs.py \
        --mask_enabled --mask_n 20 \
        --model /path/to/model --port 8000

技术细节: TRL 0.16.0 trl/scripts/vllm_serve.py 把 `app = FastAPI()` 和 `@app.post`
装饰器都放在 main() 函数内 (不是模块顶层), 所以 import 时无法 patch 路由.
解决方法是 monkey-patch uvicorn.run.
"""
import sys

# Module-level globals, 传给 closure handler (避免 closure capture vars)
_MASK_PROCESSOR_GLOBAL = None       # CVRPMaskProcessor 实例 (启用时)
_MASK_ENABLED_GLOBAL = False
_MASK_N_GLOBAL = 20
_MASK_DEBUG_GLOBAL = False


def _extract_mask_args():
    """从 sys.argv 提取 mask 相关参数, 消费后从 argv 移除.

    返回 (enabled, n, debug).
    """
    global _MASK_ENABLED_GLOBAL, _MASK_N_GLOBAL, _MASK_DEBUG_GLOBAL
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--mask_enabled":
            _MASK_ENABLED_GLOBAL = True
            i += 1
        elif arg == "--mask_n" and i + 1 < len(sys.argv):
            _MASK_N_GLOBAL = int(sys.argv[i + 1])
            i += 2
        elif arg == "--mask_debug":
            _MASK_DEBUG_GLOBAL = True
            i += 1
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv
    return _MASK_ENABLED_GLOBAL, _MASK_N_GLOBAL, _MASK_DEBUG_GLOBAL


def _install_logprobs_patch():
    """Hook uvicorn.run, 在 server 启动前替换 /generate/ 路由暴露 logprobs."""
    import uvicorn
    _orig_uvicorn_run = uvicorn.run

    def patched_uvicorn_run(app, *args, **kwargs):
        try:
            _replace_generate_route(app)
        except Exception as e:
            print(f"⚠️ logprobs patch failed (continuing without IS correction): "
                  f"{type(e).__name__}: {e}", flush=True)
        return _orig_uvicorn_run(app, *args, **kwargs)

    uvicorn.run = patched_uvicorn_run
    print("[logprobs] uvicorn.run hooked, will replace /generate/ route at startup")


def _build_mask_processor(llm_instance, n: int, debug: bool):
    """从 vLLM LLM 实例拿 tokenizer, 创建 CVRPMaskProcessor."""
    from utils.cvrp_mask_state import MaskConfig
    from utils.vllm_cvrp_mask_processor import CVRPMaskProcessor

    tokenizer = llm_instance.get_tokenizer()
    cfg = MaskConfig(enabled=True, n=n, debug_log=debug)
    proc = CVRPMaskProcessor(n=n, tokenizer=tokenizer, cfg=cfg, debug_log=debug)
    return proc


def _replace_generate_route(app):
    """从 FastAPI app 找到 /generate/ POST 路由, 替换 handler 让其返回 logprobs."""
    from typing import Optional
    from pydantic import BaseModel
    from vllm import LLM, SamplingParams

    # 1. 定位原 /generate/ 路由
    orig_route = None
    for route in app.router.routes:
        path_match = hasattr(route, "path") and route.path == "/generate/"
        methods_match = "POST" in getattr(route, "methods", set())
        if path_match and methods_match:
            orig_route = route
            break
    if orig_route is None:
        raise RuntimeError("/generate/ POST route 未找到, TRL 版本可能不兼容")

    # 2. 从原 handler closure 提取 llm 实例
    orig_handler = orig_route.endpoint
    llm_instance = None
    if orig_handler.__closure__ is not None:
        for cell in orig_handler.__closure__:
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            if isinstance(obj, LLM):
                llm_instance = obj
                break
    if llm_instance is None:
        raise RuntimeError("无法从原 /generate/ handler closure 提取 llm 实例")

    # 3. 如果启用 mask, 创建 processor 实例 (启动时一次, 多 sequence 共享)
    global _MASK_PROCESSOR_GLOBAL
    if _MASK_ENABLED_GLOBAL:
        try:
            _MASK_PROCESSOR_GLOBAL = _build_mask_processor(
                llm_instance, n=_MASK_N_GLOBAL, debug=_MASK_DEBUG_GLOBAL,
            )
            print(f"[mask] CVRPMaskProcessor 启用 (n={_MASK_N_GLOBAL}, debug={_MASK_DEBUG_GLOBAL})",
                  flush=True)
        except Exception as e:
            print(f"⚠️ [mask] CVRPMaskProcessor 创建失败, 继续不启用: "
                  f"{type(e).__name__}: {e}", flush=True)
            _MASK_PROCESSOR_GLOBAL = None
    else:
        print("[mask] CVRPMaskProcessor disabled (--mask_enabled 未传)", flush=True)

    # 4. 定义扩展 schema (加 mask_hits 字段, 即使 mask 未启用也兼容)
    class _PatchedGenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        return_logprobs: bool = False
        return_mask_hits: bool = False  # 新增: 是否返回 per-token mask_hit

    class _PatchedGenerateResponse(BaseModel):
        completion_ids: list[list[int]]
        logprobs: Optional[list[list[float]]] = None
        mask_hits: Optional[list[list[bool]]] = None  # 新增

    # 5. 注销原路由
    app.router.routes.remove(orig_route)

    # 6. 注册新路由
    @app.post("/generate/", response_model=_PatchedGenerateResponse)
    async def generate(request: _PatchedGenerateRequest):
        # Guided decoding (跟 TRL 0.16 原 handler 一致)
        guided_decoding = None
        if request.guided_decoding_regex is not None:
            from vllm.sampling_params import GuidedDecodingParams
            guided_decoding = GuidedDecodingParams(
                backend="outlines", regex=request.guided_decoding_regex
            )

        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
            logprobs=1 if request.return_logprobs else None,
        )

        # 实例字段赋值注入 mask processor (msgspec.Struct __init__ 不能 patch,
        # 调研结论 + probe 验证)
        if _MASK_PROCESSOR_GLOBAL is not None:
            sampling_params.logits_processors = [_MASK_PROCESSOR_GLOBAL]

        all_outputs = llm_instance.generate(
            request.prompts, sampling_params=sampling_params
        )

        completion_ids: list[list[int]] = [
            list(output.token_ids)
            for outputs in all_outputs
            for output in outputs.outputs
        ]

        # ── logprobs (post-mask 来自 vLLM sampler) ──────────────────
        logprobs_out: Optional[list[list[float]]] = None
        if request.return_logprobs:
            logprobs_out = []
            for outputs in all_outputs:
                for output in outputs.outputs:
                    per_token_lp: list[float] = []
                    if output.logprobs is not None:
                        for t in range(len(output.token_ids)):
                            if t >= len(output.logprobs):
                                per_token_lp.append(0.0)
                                continue
                            lp_dict = output.logprobs[t]
                            if lp_dict is None:
                                per_token_lp.append(0.0)
                                continue
                            sampled_id = output.token_ids[t]
                            lp_obj = lp_dict.get(sampled_id)
                            per_token_lp.append(
                                float(lp_obj.logprob) if lp_obj is not None else 0.0
                            )
                    logprobs_out.append(per_token_lp)

        # ── mask_hits (post-hoc 重建每个 token 是否被 mask 触发) ─────
        # 用同样的 state machine, 模拟 token-by-token 推进
        mask_hits_out: Optional[list[list[bool]]] = None
        if request.return_mask_hits and _MASK_PROCESSOR_GLOBAL is not None:
            mask_hits_out = _post_process_mask_hits(
                request.prompts, all_outputs, _MASK_PROCESSOR_GLOBAL,
            )

        return {
            "completion_ids": completion_ids,
            "logprobs": logprobs_out,
            "mask_hits": mask_hits_out,
        }

    print(f"[logprobs] /generate/ replaced (return_logprobs / return_mask_hits 双开关; "
          f"mask processor: {'enabled' if _MASK_PROCESSOR_GLOBAL else 'disabled'})",
          flush=True)


def _post_process_mask_hits(prompts, all_outputs, mask_processor):
    """事后重建每个 token 位置是否触发 mask.

    对每个完成的 sequence, 模拟从 0 开始 token-by-token decode,
    在每个位置跑 state machine + compute_mask, 标记 mask_hit.

    O(L) per sequence (增量推进).
    """
    from utils.cvrp_mask_state import build_state, compute_mask

    tokenizer = mask_processor.tokenizer
    cfg = mask_processor.cfg
    mask_hits_all = []

    for prompt, outputs in zip(prompts, all_outputs):
        for output in outputs.outputs:
            token_ids = list(output.token_ids)
            mask_hits = []
            for t in range(len(token_ids)):
                # 生成 token t 之前的状态
                past_ids = token_ids[:t]
                past_text = (
                    tokenizer.decode(past_ids, skip_special_tokens=False)
                    if past_ids else ""
                )
                state = build_state(past_text, n=cfg.n)
                decision = compute_mask(state, cfg)
                mask_hits.append(bool(decision.mask_hit))
            mask_hits_all.append(mask_hits)

    return mask_hits_all


if __name__ == "__main__":
    # 提取 mask CLI 参数 (消费后从 sys.argv 移除, 不传给 trl)
    enabled, n, debug = _extract_mask_args()
    print(f"[init] mask_enabled={enabled}, mask_n={n}, mask_debug={debug}", flush=True)

    # 在 runpy 之前 hook uvicorn.run, 这样 TRL main() 调 uvicorn.run 时我们抢先替换路由
    _install_logprobs_patch()

    # TRL 0.16.0 的 vllm_serve 入口
    import runpy

    runpy.run_module("trl.scripts.vllm_serve", run_name="__main__")
