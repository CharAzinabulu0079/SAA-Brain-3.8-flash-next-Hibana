#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible single-stream and concurrent throughput benchmark for Hibana."""

import argparse
import asyncio
import json
import time

import httpx


async def run_stream(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int, temp: float, headers: dict):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    t_first = t_last = None
    chunks = 0
    usage = None
    async with client.stream("POST", url, json=body, headers=headers) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            obj = json.loads(line[5:])
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if (ch.get("delta") or {}).get("content"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks += 1
    out_toks = (usage or {}).get("completion_tokens") or chunks
    ttft = (t_first - t0) if t_first else 0.0
    decode_s = max(1e-6, (t_last - t_first) if (t_first and t_last) else (time.perf_counter() - t0))
    return out_toks, ttft, decode_s, time.perf_counter() - t0


async def main_async(args: argparse.Namespace) -> None:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    prompts = [
        ("zh_prose", "用清晰有画面感的中文散文描写深夜雨后城市高架桥上的车流与光影。", 0.0),
        ("en_prose", "Write a vivid 400-word essay about midnight rain in Shanghai.", 0.0),
        ("zh_chat_t07", "刚忙完一天回家累得不行，用自然口语和我聊几句放松一下，别说套话。", 0.7),
        ("code", "Write a complete async Python token-bucket rate limiter with unit tests.", 0.0),
    ]
    async with httpx.AsyncClient(timeout=180.0) as client:
        print("=== Single-Stream Benchmark ===")
        for name, prompt, temp in prompts:
            toks, ttft, dec_s, _ = await run_stream(client, args.url, args.model, prompt, args.max_tokens, temp, headers)
            print(f"{name:14s}: tok={toks:4d} ttft={ttft:.2f}s decode_tps={max(1, toks - 1)/dec_s:.2f} tok/s")

        if args.concurrency > 1:
            print(f"\n=== {args.concurrency}-Way Concurrent Benchmark (en_prose) ===")
            t0 = time.perf_counter()
            res = await asyncio.gather(*[
                run_stream(client, args.url, args.model, prompts[1][1], args.max_tokens, 0.0, headers)
                for _ in range(args.concurrency)
            ])
            wall = time.perf_counter() - t0
            tot_toks = sum(r[0] for r in res)
            print(f"ok={len(res)}/{args.concurrency} total_toks={tot_toks} wall={wall:.2f}s agg={tot_toks/wall:.1f} tok/s per_stream={tot_toks/wall/args.concurrency:.1f} tok/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30004/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
