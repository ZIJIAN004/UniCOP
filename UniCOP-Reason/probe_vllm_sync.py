"""
探针：定期打 vLLM server /generate，看同一 prompt 的输出是否随训练变化
用法：
    python probe_vllm_sync.py --port 8001 --interval 60 --rounds 10

解释：
    固定 prompt + seed + temperature=0（贪心），理论上"同一模型"会给完全相同输出。
    如果训练过程中 vLLM 的权重被正确同步，各轮输出应该逐渐变化。
    如果各轮输出**完全一致**，说明 vLLM 根本没收到更新，权重同步失败。
"""
import argparse
import hashlib
import json
import time
import urllib.request


def fetch(url: str, payload: dict, timeout=60) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def probe(host: str, port: int, prompt: str) -> tuple[str, str]:
    """返回 (raw_response_hash, 前 50 token 字符串)。"""
    url = f"http://{host}:{port}/generate"
    payload = {
        "prompts": [prompt],
        "n": 1,
        "max_tokens": 64,
        "temperature": 0.0,        # 贪心，确定性
        "seed": 42,
    }
    data = fetch(url, payload)
    # TRL vllm-serve 的返回字段名在不同版本可能叫 completion_ids / completions / tokens
    # 这里 best-effort 提取
    raw = json.dumps(data, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    # 尝试几种常见字段名
    preview = ""
    for key in ("completions", "completion_ids", "tokens", "text", "outputs"):
        if key in data:
            val = data[key]
            if isinstance(val, list) and val:
                preview = str(val[0])[:80]
                break
    return digest, preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True,
                    help="vLLM server 端口（见 auto_train.sh 里的 VLLM_PORT_BASE + task_idx）")
    ap.add_argument("--interval", type=int, default=60,
                    help="每轮间隔秒数（默认 60 秒一次）")
    ap.add_argument("--rounds", type=int, default=10,
                    help="总探测轮数")
    ap.add_argument("--prompt", default="Solve TSP with 3 cities at (0,0), (1,0), (0,1). Answer:",
                    help="用于探测的固定 prompt")
    args = ap.parse_args()

    print(f"探针目标: http://{args.host}:{args.port}/generate")
    print(f"Prompt:  {args.prompt}")
    print(f"间隔:    {args.interval}s × {args.rounds} 轮\n")

    history = []
    for i in range(args.rounds):
        try:
            digest, preview = probe(args.host, args.port, args.prompt)
            history.append(digest)
            tag = "DIFF" if len(history) == 1 or digest != history[-2] else "SAME"
            print(f"[Round {i+1:02d}] hash={digest}  [{tag}]  preview={preview[:60]}")
        except Exception as e:
            print(f"[Round {i+1:02d}] 请求失败: {e}")
        if i < args.rounds - 1:
            time.sleep(args.interval)

    print("\n" + "=" * 60)
    unique = set(history)
    print(f"探测 {len(history)} 轮，唯一输出 {len(unique)} 种")
    if len(unique) == 1:
        print("[BROKEN] 所有轮输出完全一致 → vLLM 权重从未更新，sync 失败")
        print("         可能原因：")
        print("           - vLLM 版本不兼容（如 0.17.0 的 Qwen 同步 bug, trl#5269）")
        print("           - TRL server 模式的 _move_model_to_vllm 未触发")
        print("           - 训练还没跑到第一个 gradient update（max_steps 太少？）")
    elif len(unique) < len(history) // 2:
        print("[SUSPECT] 唯一输出过少，权重可能只同步了一次或很少几次")
    else:
        print("[OK] 输出随训练变化，权重同步正常")


if __name__ == "__main__":
    main()
