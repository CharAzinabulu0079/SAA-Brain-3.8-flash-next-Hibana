# Reverse-Proxy QoS & Multi-Agent Concurrency Pattern

When serving a single 121.7 GiB GB10 unified-memory node across real-time voice/interactive clients, IDE coding agents (with 10–16 parallel subagents), and background batch workloads, placing a lightweight priority-queue reverse proxy in front of `127.0.0.1:30004` prevents head-of-line blocking without increasing engine memory.

## Key Design Points

1. **Loopback Engine Isolation**:
   - Bind Hibana exclusively to `127.0.0.1:30004` (`HIBANA_HOST=127.0.0.1`).
   - Expose only the authenticated reverse proxy on LAN/external interfaces.

2. **Priority Slots Aligned with `HIBANA_MAX_BATCH=16`**:
   - Set `QOS_SLOTS = 17` (`16` shared seats matching `HIBANA_MAX_BATCH=16`, plus `1` seat strictly reserved for `P0` real-time voice/interactive priority).
   - Set `QOS_QUEUE_CAP = 16` so burst requests wait cleanly in lightweight proxy/scheduler threads instead of triggering extra KV/state-pool allocations.

3. **Session-Affinity Header Parsing & Subagent Concurrency**:
   - Multi-agent harnesses (such as OpenAI-compatible clients using `pi-ai`) emit `x-session-affinity` in addition to `x-session-id` / `x-client-session`.
   - If your proxy applies per-session concurrency caps (`per_session_cap`) for long-context agent turns (`max_tokens > 8192`), parse `x-session-affinity` in the session key extractor and allow an explicit per-client ceiling (e.g. `16`) for interactive agent/benchmark keys while keeping background/learning keys capped at `1`.
