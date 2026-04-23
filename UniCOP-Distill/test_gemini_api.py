"""Gemini API 单次请求诊断脚本，用于排查 rate limit / 认证 / 配额问题。"""
import os
import sys
import time

CREDENTIAL_PATH = "/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/advance-subject-493905-h9-020e2dc30ae7.json"
PROJECT = "advance-subject-493905-h9"
LOCATION = "us-central1"
MODEL = "gemini-2.5-pro"


def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL_PATH

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("[FAIL] google-genai 未安装，请运行: pip install google-genai")
        sys.exit(1)

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    print(f"Credential : {CREDENTIAL_PATH}")
    print(f"Project    : {PROJECT}")
    print(f"Location   : {LOCATION}")
    print(f"Model      : {MODEL}")
    print("-" * 50)

    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents="Say hello in one sentence.",
            config=types.GenerateContentConfig(max_output_tokens=32),
        )
        elapsed = time.time() - t0
        print(f"[OK] {elapsed:.2f}s")
        print(f"Response: {resp.text}")
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            um = resp.usage_metadata
            print(f"Tokens — prompt: {um.prompt_token_count}, "
                  f"completion: {um.candidates_token_count}, "
                  f"total: {um.total_token_count}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAIL] {elapsed:.2f}s")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
