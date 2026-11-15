# Security Policy

## Reporting & Secret Hygiene

This repository contains **configuration templates, patches, and training/benchmarking scripts only**. It contains **zero credentials, API keys, hostnames, private IP addresses, or model weights**.

## Production Deployment Guidelines

1. **Loopback-Only Engine Binding**: Always bind the inference engine (`HIBANA_HOST=127.0.0.1`) to loopback only. Never expose an unauthenticated inference engine port directly to a LAN or public interface.
2. **Authenticated Reverse Proxy**: Place an authenticated HTTP reverse proxy in front of the loopback engine for external or multi-client access, enforcing per-client API keys and priority-based QoS admission control.
3. **Unified-Memory Guard (`HIBANA_HOST_FLOOR_GIB`)**: On single unified-memory systems (such as NVIDIA DGX Spark GB10, 121.7 GiB), keep `HIBANA_HOST_FLOOR_GIB` configured (e.g., `4` GiB, which sets the `memory guard` tripwire at `2` GiB `MemAvailable`) so concurrent media/vision workloads cannot starve the OS.
