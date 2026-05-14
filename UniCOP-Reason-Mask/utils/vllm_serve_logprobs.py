"""
Wrapper: 替换 /generate/ 路由暴露 sampled-token logprobs, 然后启动 trl vllm-serve.

仅一个 patch:
  /generate/ endpoint 返回 sampled-token logprob (for GRPO Importance Sampling 校正):
    - 扩展 GenerateRequest 加 return_logprobs: bool 字段
    - 扩展 GenerateResponse 加 logprobs: Optional[list[list[float]]] 字段
    - 替换 handler 内 SamplingParams 时设 logprobs=1, 从 output 取 sampled token logprob

  这样训练端能算 IS ratio = exp(train_logps - vllm_logps), 修正 vLLM 跟训练
  attention kernel 数值差异造成的 GRPO policy gradient 偏差.

(NoRepeatNgram patch 已删除: 用户决定不用 ngram 抑制, 改用 CVRP-specific mask processor.
 Mask processor 的集成见 vllm_cvrp_mask_processor.py 和 Phase 3 改动.)

技术细节: TRL 0.16.0 trl/scripts/vllm_serve.py 把 `app = FastAPI()` 和 `@app.post`
装饰器都放在 main() 函数内 (不是模块顶层), 所以 import 时无法 patch 路由.
解决方法是 monkey-patch uvicorn.run: 在 main() 内 uvicorn.run(app, ...) 真正
启动 server 之前, 拿到 app 实例, 从原 /generate/ handler 的 __closure__ 提取 llm 实例,
然后注销原路由 + 注册新路由 (新 handler 用提取到的 llm 闭包).

用法 (替代 trl vllm-serve):
    python utils/vllm_serve_logprobs.py \
        --model /path/to/model \
        --port 8000 \
        [其他 trl vllm-serve 参数原样传递]
"""


def _install_logprobs_patch():
    """Hook uvicorn.run, 在 server 启动前替换 /generate/ 路由暴露 logprobs.

    流程:
      main() 内顺序: llm = LLM(...) → app = FastAPI() → @app.post(...) 注册 handler
                  → uvicorn.run(app, ...)  ← 我们在这里抢先 hook
      hook 内: 找 app /generate/ POST 路由, 从原 handler.__closure__ 提取 llm,
               注销原路由, 注册新 handler (logprobs-aware).
    """
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

    # 2. 从原 handler closure 提取 llm 实例 (LLM 类型识别)
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

    # 3. 定义扩展 schema
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
        return_logprobs: bool = False  # 新增

    class _PatchedGenerateResponse(BaseModel):
        completion_ids: list[list[int]]
        logprobs: Optional[list[list[float]]] = None  # 新增

    # 4. 注销原路由
    app.router.routes.remove(orig_route)

    # 5. 注册新路由
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

        all_outputs = llm_instance.generate(
            request.prompts, sampling_params=sampling_params
        )

        completion_ids: list[list[int]] = [
            list(output.token_ids)
            for outputs in all_outputs
            for output in outputs.outputs
        ]

        if not request.return_logprobs:
            return {"completion_ids": completion_ids, "logprobs": None}

        # 提取 sampled token 的 logprob: output.logprobs[t] is dict[int, Logprob]
        # logprobs=1 时该 dict 至少包含 sampled token (key=token_ids[t]).
        logprobs_out: list[list[float]] = []
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

        return {"completion_ids": completion_ids, "logprobs": logprobs_out}

    print(f"[logprobs] /generate/ replaced (return_logprobs=True 时返回 per-token "
          f"sampled logprobs, llm extracted: {type(llm_instance).__name__})",
          flush=True)


if __name__ == "__main__":
    # 在 runpy 之前 hook uvicorn.run, 这样 TRL main() 调 uvicorn.run 时我们抢先替换路由
    _install_logprobs_patch()

    # TRL 0.16.0 的 vllm_serve 入口
    import runpy

    runpy.run_module("trl.scripts.vllm_serve", run_name="__main__")
