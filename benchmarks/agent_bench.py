#!/usr/bin/env python3
"""
Multi-agent context-heavy benchmark for OpenAI-compatible endpoints.

Design:
- 1 main agent + 0..4 subagents (total 1,2,3,4,5 agents)
- Each agent shares a long common prefix (~70% of prompt tokens) to hit prefix cache
- Each agent has unique recent conversation turns (~30% of prompt tokens)
- Context grows until it reaches ~60% of max_model_len, then a compaction turn
  sends a "summarize" request and resets history to [system, summary].
- Some turns include a tiny base64 image to exercise multimodal path.
- Engines are tested serially (one at a time) to keep GPU measurements clean.

Metrics recorded:
- TTFT (time to first token) per request
- Strict decode throughput (tok/s) per request: completion_tokens / (last_token_time - first_token_time)
- End-to-end streaming throughput (tok/s) per request: completion_tokens / total_latency
- End-to-end latency per request
- Total throughput across agents
- Warmup vs steady-state splits
- Cache hit rate (vLLM logs or approximated)
- Context length at each turn
- Compaction events

Important distinctions:
- "decode tok/s" excludes TTFT and network/request overhead; it measures pure token generation speed.
- "e2e tok/s" includes TTFT, request serialization, and streaming overhead; it reflects what a browser/client sees.
- First-request warmup on V100/SM70 is slow and is reported separately from steady-state metrics.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# A tiny valid 1x1 red PNG for multimodal turns
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def now() -> float:
    return time.perf_counter()


def estimate_tokens(text: str) -> int:
    # Very rough estimate: ~0.3 tokens per char for English/Chinese mix
    return max(1, int(len(text) * 0.3))


def call_chat(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout: int = 300,
    stream: bool = True,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = now()
    response = urllib.request.urlopen(req, timeout=timeout)
    first_token_time: float | None = None
    content_chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    if stream:
        for raw in response:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[6:]
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Usage-only chunk has empty choices
            choices = chunk.get("choices")
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = now()
                    content_chunks.append(content)
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
        if first_token_time is None:
            first_token_time = start
    else:
        data = json.loads(response.read())
        content_chunks = [data["choices"][0]["message"]["content"]]
        first_token_time = start
        prompt_tokens = data["usage"]["prompt_tokens"]
        completion_tokens = data["usage"]["completion_tokens"]

    end = now()
    content = "".join(content_chunks)
    # Fallback token estimation for endpoints that omit usage
    if completion_tokens == 0:
        completion_tokens = max(1, estimate_tokens(content))
    if prompt_tokens == 0:
        prompt_tokens = max(1, sum(estimate_tokens(str(m.get("content", ""))) for m in messages))

    latency = end - start
    ttft = first_token_time - start
    decode_time = max(end - first_token_time, 1e-6)
    decode_tok_per_sec = completion_tokens / decode_time
    e2e_tok_per_sec = completion_tokens / max(latency, 1e-6)

    return {
        "content": content,
        "latency": latency,
        "ttft": ttft,
        "decode_time": decode_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tok_per_sec": decode_tok_per_sec,
        "e2e_tok_per_sec": e2e_tok_per_sec,
    }


@dataclass
class TurnResult:
    agent_id: int
    turn: int
    compacted: bool
    multimodal: bool
    prompt_tokens: int
    completion_tokens: int
    latency: float
    ttft: float
    decode_tok_per_sec: float
    e2e_tok_per_sec: float
    context_tokens_before: int
    context_tokens_after: int
    warmup: bool = False


@dataclass
class AgentRun:
    agent_id: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    context_tokens: int = 0
    compact_count: int = 0
    results: list[TurnResult] = field(default_factory=list)

    def reset_with_summary(self, summary: str, system: dict[str, Any]) -> None:
        self.messages = [system, {"role": "assistant", "content": summary}]
        self.context_tokens = estimate_tokens(system["content"]) + estimate_tokens(summary)
        self.compact_count += 1


def build_shared_prefix(target_tokens: int) -> str:
    """Build a long shared text block of approximately target_tokens tokens."""
    # Each paragraph is ~150 tokens
    paragraphs = [
        "You are an expert research assistant. You have access to a large corpus of technical documents. "
        "Your task is to answer questions accurately, cite sources when possible, and maintain a consistent "
        "tone throughout the conversation. Always reason step by step before giving the final answer.",
        "Project guidelines: prioritize correctness over speed. If uncertain, state the uncertainty clearly. "
        "Use the provided context documents as the primary source of truth. Do not hallucinate facts.",
        "Tooling conventions: when calling functions, emit well-formed XML. Validate arguments before use. "
        "Report errors concisely and suggest a fix when a tool call fails.",
    ]
    corpus_sentences = [
        "The transformer architecture relies on self-attention to model long-range dependencies.",
        "Mixture-of-experts models route tokens to specialized sub-networks to reduce compute.",
        "Quantization maps high-precision weights to lower bit-widths to save memory and bandwidth.",
        "FlashAttention reorders attention computation to reduce HBM traffic on GPUs.",
        "Speculative decoding uses a small draft model to predict tokens verified by the target model.",
        "Linear attention approximates softmax attention with kernel feature maps for faster decoding.",
        "KV cache stores intermediate key and value tensors to avoid recomputation during generation.",
        "Context compression summarizes long conversations into a shorter working memory buffer.",
        "Prefix caching reuses computed attention states for prompts sharing a common prefix.",
        "Group-query attention shares key/value heads across query heads to shrink cache size.",
        "AWQ quantizes weights based on activation-aware scaling to preserve accuracy.",
        "GPTQ uses approximate second-order information for one-shot weight quantization.",
        "SmoothQuant migrates quantization difficulty from activations to weights via scaling.",
        "Tensor parallelism shards layers across GPUs to fit large models on multiple devices.",
        "Pipeline parallelism splits the model depth across stages to increase throughput.",
        "Continuous batching improves GPU utilization by dynamically assembling request batches.",
        "PageAttention stores KV cache in non-contiguous blocks to reduce memory fragmentation.",
        "LoRA adds low-rank adapters to fine-tune large models with minimal trainable parameters.",
        "RLHF aligns language models with human preferences using reward models and PPO.",
        "DPO directly optimizes the policy against pairwise preference data without a reward model.",
    ]
    text = "\n\n".join(paragraphs) + "\n\n"
    current = estimate_tokens(text)
    idx = 0
    while current < target_tokens and idx < 10000:
        sentence = corpus_sentences[idx % len(corpus_sentences)]
        addition = f"Document {idx}: {sentence} "
        text += addition
        current += estimate_tokens(addition)
        idx += 1
    return text


def build_unique_query(agent_id: int, turn: int, mode: str) -> str:
    queries = {
        "analysis": [
            f"Agent-{agent_id}: Summarize the key trade-offs between accuracy and latency in the documents.",
            f"Agent-{agent_id}: Which quantization method is best for a 27B model on a 32GB GPU?",
            f"Agent-{agent_id}: Explain how speculative decoding interacts with prefix caching.",
            f"Agent-{agent_id}: Compare linear attention and full attention for long-context retrieval.",
            f"Agent-{agent_id}: List three ways to reduce KV cache memory footprint.",
        ],
        "multimodal": [
            f"Agent-{agent_id}: Describe the image and relate it to the technical concepts above.",
            f"Agent-{agent_id}: What does this visual example illustrate about model compression?",
        ],
        "compact": [
            f"Agent-{agent_id}: Produce a concise summary of everything we have discussed so far, "
            "preserving all facts and action items. Keep it under 200 words.",
        ],
    }
    pool = queries.get(mode, queries["analysis"])
    return pool[(agent_id + turn) % len(pool)]


def make_image_content() -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{TINY_PNG_B64}"},
    }


def run_one_turn(
    agent: AgentRun,
    endpoint: str,
    model: str,
    shared_system: dict[str, Any],
    shared_prefix_text: str,
    compact_threshold: int,
    max_model_len: int,
    max_tokens: int,
    force_compact: bool = False,
    force_multimodal: bool = False,
) -> TurnResult:
    agent.turn += 1
    compacted = False

    # Decide if compaction needed
    if force_compact or agent.context_tokens >= compact_threshold:
        # Summarize
        summary_query = build_unique_query(agent.agent_id, agent.turn, "compact")
        agent.messages.append({"role": "user", "content": summary_query})
        resp = call_chat(endpoint, model, agent.messages, max_tokens=max_tokens)
        summary = resp["content"]
        ctx_before = agent.context_tokens
        agent.reset_with_summary(summary, shared_system)
        ctx_after = agent.context_tokens
        compacted = True
        return TurnResult(
            agent_id=agent.agent_id,
            turn=agent.turn,
            compacted=True,
            multimodal=False,
            prompt_tokens=resp["prompt_tokens"],
            completion_tokens=resp["completion_tokens"],
            latency=resp["latency"],
            ttft=resp["ttft"],
            decode_tok_per_sec=resp["decode_tok_per_sec"],
            e2e_tok_per_sec=resp["e2e_tok_per_sec"],
            context_tokens_before=ctx_before,
            context_tokens_after=ctx_after,
        )

    # Normal turn
    is_mm = force_multimodal or (agent.turn % 5 == 0)
    query = build_unique_query(agent.agent_id, agent.turn, "multimodal" if is_mm else "analysis")
    if is_mm:
        user_content = [make_image_content(), {"type": "text", "text": query}]
    else:
        user_content = query
    agent.messages.append({"role": "user", "content": user_content})

    ctx_before = agent.context_tokens
    resp = call_chat(endpoint, model, agent.messages, max_tokens=max_tokens)
    agent.messages.append({"role": "assistant", "content": resp["content"]})
    # Re-estimate context after adding assistant response
    agent.context_tokens = (
        estimate_tokens(shared_prefix_text)
        + sum(estimate_tokens(str(m.get("content", ""))) for m in agent.messages[1:])
    )
    ctx_after = agent.context_tokens

    return TurnResult(
        agent_id=agent.agent_id,
        turn=agent.turn,
        compacted=False,
        multimodal=is_mm,
        prompt_tokens=resp["prompt_tokens"],
        completion_tokens=resp["completion_tokens"],
        latency=resp["latency"],
        ttft=resp["ttft"],
        decode_tok_per_sec=resp["decode_tok_per_sec"],
        e2e_tok_per_sec=resp["e2e_tok_per_sec"],
        context_tokens_before=ctx_before,
        context_tokens_after=ctx_after,
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return sorted(values)[f]
    return sorted(values)[f] * (c - k) + sorted(values)[c] * (k - f)


def summarize_results(
    all_results: list[TurnResult],
    wall_elapsed: float,
    num_agents: int,
    max_model_len: int,
    shared_prefix_tokens: int,
    turns_per_agent: int,
    warmup_turns: int,
) -> dict[str, Any]:
    """Compute aggregate metrics and split warmup from steady-state."""
    for r in all_results:
        r.warmup = r.turn <= warmup_turns

    steady_results = [r for r in all_results if not r.warmup]
    warmup_results = [r for r in all_results if r.warmup]

    total_completion_tokens = sum(r.completion_tokens for r in all_results)
    total_prompt_tokens = sum(r.prompt_tokens for r in all_results)
    compact_count = sum(1 for r in all_results if r.compacted)
    mm_count = sum(1 for r in all_results if r.multimodal)

    avg_prompt_tokens = total_prompt_tokens / len(all_results) if all_results else 1
    approx_cache_hit = min(0.99, shared_prefix_tokens / avg_prompt_tokens) if avg_prompt_tokens else 0

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "avg": statistics.mean(values) if values else 0.0,
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    all_latencies = [r.latency for r in all_results]
    all_ttfts = [r.ttft for r in all_results]
    all_decode_tps = [r.decode_tok_per_sec for r in all_results]
    all_e2e_tps = [r.e2e_tok_per_sec for r in all_results]

    steady_latencies = [r.latency for r in steady_results]
    steady_ttfts = [r.ttft for r in steady_results]
    steady_decode_tps = [r.decode_tok_per_sec for r in steady_results]
    steady_e2e_tps = [r.e2e_tok_per_sec for r in steady_results]
    steady_completion_tokens = sum(r.completion_tokens for r in steady_results)
    # Estimate steady-state wall time by excluding warmup turns' share of total wall time.
    # This is an approximation because warmup and steady turns are not strictly serial.
    steady_wall_time = wall_elapsed * (len(steady_results) / max(len(all_results), 1)) if all_results else 1e-6

    return {
        "num_agents": num_agents,
        "max_model_len": max_model_len,
        "compact_threshold": int(max_model_len * 0.6),
        "turns_per_agent": turns_per_agent,
        "warmup_turns": warmup_turns,
        "total_requests": len(all_results),
        "warmup_requests": len(warmup_results),
        "steady_state_requests": len(steady_results),
        "wall_time_sec": wall_elapsed,
        "total_throughput_tok_per_sec": total_completion_tokens / wall_elapsed,
        "all_requests": {
            "latency_sec": stats(all_latencies),
            "ttft_sec": stats(all_ttfts),
            "decode_tok_per_sec": stats(all_decode_tps),
            "e2e_tok_per_sec": stats(all_e2e_tps),
        },
        "steady_state": {
            "latency_sec": stats(steady_latencies),
            "ttft_sec": stats(steady_ttfts),
            "decode_tok_per_sec": stats(steady_decode_tps),
            "e2e_tok_per_sec": stats(steady_e2e_tps),
            "total_completion_tokens": steady_completion_tokens,
            "total_throughput_tok_per_sec": steady_completion_tokens / max(steady_wall_time, 1e-6),
        },
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "compact_count": compact_count,
        "multimodal_count": mm_count,
        "approx_cache_hit_rate": approx_cache_hit,
        "results": all_results,
    }


def run_agents(
    endpoint: str,
    model: str,
    num_agents: int,
    max_model_len: int,
    shared_prefix_tokens: int,
    turns_per_agent: int,
    max_tokens: int,
    compact_ratio: float = 0.6,
    warmup_turns: int = 1,
) -> dict[str, Any]:
    compact_threshold = int(max_model_len * compact_ratio)
    shared_prefix_text = build_shared_prefix(shared_prefix_tokens)
    shared_system = {"role": "system", "content": shared_prefix_text}

    agents: list[AgentRun] = []
    for i in range(num_agents):
        ar = AgentRun(agent_id=i)
        ar.messages = [shared_system]
        ar.context_tokens = estimate_tokens(shared_prefix_text)
        agents.append(ar)

    all_results: list[TurnResult] = []
    wall_start = now()

    for turn_idx in range(turns_per_agent):
        # Run all agents concurrently for this turn
        threads: list[threading.Thread] = []
        per_agent_results: list[TurnResult] = []
        lock = threading.Lock()

        def worker(agent: AgentRun) -> None:
            # 30% of each prompt is unique recent conversation; shared prefix is cached
            result = run_one_turn(
                agent,
                endpoint,
                model,
                shared_system,
                shared_prefix_text,
                compact_threshold,
                max_model_len,
                max_tokens,
            )
            with lock:
                per_agent_results.append(result)

        for agent in agents:
            t = threading.Thread(target=worker, args=(agent,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        all_results.extend(per_agent_results)
        for r in per_agent_results:
            agents[r.agent_id].results.append(r)

    wall_elapsed = now() - wall_start

    return summarize_results(
        all_results,
        wall_elapsed,
        num_agents,
        max_model_len,
        shared_prefix_tokens,
        turns_per_agent,
        warmup_turns,
    )


def wait_for_health(endpoint: str, timeout: int = 300) -> bool:
    deadline = now() + timeout
    while now() < deadline:
        try:
            urllib.request.urlopen(f"{endpoint}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def stop_all_servers() -> None:
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], capture_output=True)
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)


def capture_environment() -> dict[str, Any]:
    """Capture basic environment details for reproducibility."""
    env: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": sys.platform,
    }
    try:
        import torch

        env["pytorch_version"] = torch.__version__
        env["pytorch_cuda_version"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_compute_capability"] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
            env["gpu_total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                env["nvidia_smi"] = {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "compute_capability": parts[3],
                }
    except Exception:
        pass
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-27b-awq")
    parser.add_argument("--num-agents", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--max-model-len", type=int, default=51744)
    parser.add_argument("--shared-prefix-tokens", type=int, default=15000)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--compact-ratio", type=float, default=0.6)
    parser.add_argument("--warmup-turns", type=int, default=1, help="Number of initial turns per agent to discard as warmup")
    parser.add_argument("--output", default="/tmp/agent_bench_result.json")
    parser.add_argument("--wait-for-server", action="store_true")
    args = parser.parse_args()

    if args.wait_for_server:
        if not wait_for_health(args.endpoint):
            print(f"Server {args.endpoint} did not become healthy", file=sys.stderr)
            return 1

    print(f"Benchmarking {args.endpoint} with {args.num_agents} agent(s), max_model_len={args.max_model_len}")
    summary = run_agents(
        args.endpoint,
        args.model,
        args.num_agents,
        args.max_model_len,
        args.shared_prefix_tokens,
        args.turns,
        args.max_tokens,
        args.compact_ratio,
        args.warmup_turns,
    )
    summary["environment"] = capture_environment()
    summary["benchmark_config"] = {
        "endpoint": args.endpoint,
        "model": args.model,
        "num_agents": args.num_agents,
        "max_model_len": args.max_model_len,
        "shared_prefix_tokens": args.shared_prefix_tokens,
        "turns_per_agent": args.turns,
        "max_tokens": args.max_tokens,
        "compact_ratio": args.compact_ratio,
        "warmup_turns": args.warmup_turns,
    }

    # Strip heavy per-turn list from JSON summary; keep raw in a separate file if needed
    raw_results = summary.pop("results")
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    raw_path = args.output.replace(".json", "_raw.json")
    with open(raw_path, "w") as f:
        json.dump(
            [
                {
                    "agent_id": r.agent_id,
                    "turn": r.turn,
                    "warmup": r.warmup,
                    "compacted": r.compacted,
                    "multimodal": r.multimodal,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "latency": r.latency,
                    "ttft": r.ttft,
                    "decode_tok_per_sec": r.decode_tok_per_sec,
                    "e2e_tok_per_sec": r.e2e_tok_per_sec,
                    "ctx_before": r.context_tokens_before,
                    "ctx_after": r.context_tokens_after,
                }
                for r in raw_results
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary written to {args.output}")
    print(f"Raw results written to {raw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
