"""Gemini API 诊断脚本：单次测试 + RPM 实测模式。"""
import os
import sys
import time

CREDENTIAL_PATH = "/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/advance-subject-493905-h9-020e2dc30ae7.json"
PROJECT = "advance-subject-493905-h9"
LOCATION = "us-central1"
MODEL = "gemini-2.5-pro"


def single_test(client, types):
    print("=== 单次请求测试 ===")
    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents="Say hello in one sentence.",
            config=types.GenerateContentConfig(max_output_tokens=32),
        )
        print(f"[OK] {time.time() - t0:.2f}s — {resp.text}")
    except Exception as e:
        print(f"[FAIL] {time.time() - t0:.2f}s — {e}")
        sys.exit(1)


def rpm_test(client, types, n=20):
    print(f"\n=== RPM 实测：连续发 {n} 个短请求 ===")
    results = []
    for i in range(1, n + 1):
        t0 = time.time()
        try:
            client.models.generate_content(
                model=MODEL,
                contents=f"Say the number {i}.",
                config=types.GenerateContentConfig(max_output_tokens=16),
            )
            elapsed = time.time() - t0
            results.append(("OK", elapsed))
            print(f"  [{i:2d}/{n}] OK   {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            err = str(e)
            status = "429" if "429" in err or "RESOURCE_EXHAUSTED" in err else "ERR"
            results.append((status, elapsed))
            print(f"  [{i:2d}/{n}] {status}  {elapsed:.2f}s — {err[:120]}")
            if status == "429":
                print(f"\n>>> 第 {i} 个请求被限流，实际 RPM 约为 {i - 1}")
                break

    ok_count = sum(1 for s, _ in results if s == "OK")
    total_time = sum(t for _, t in results)
    print(f"\n结果: {ok_count}/{len(results)} 成功, 总耗时 {total_time:.1f}s")
    if ok_count == n:
        print(f"全部通过，RPM >= {n}")


def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL_PATH

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("[FAIL] google-genai 未安装，请运行: pip install google-genai")
        sys.exit(1)

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    print(f"Project: {PROJECT} | Location: {LOCATION} | Model: {MODEL}")
    print("-" * 50)

    single_test(client, types)
    rpm_test(client, types)


if __name__ == "__main__":
    main()
