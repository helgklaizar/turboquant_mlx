# Architecture TurboQuant Mac

Objective: Implement QJL and PolarQuant algorithms for unprecedented LLM KV Cache compression (down to 3 bits) without quality loss.
- `mlx_core/mlx_turboquant.py` — Metal-optimized full quantization pipeline (Keys)
- `mlx_core/mlx_polarquant.py` — Fast MSE quantization (Values)
- `mlx_core/cache.py` — Dynamic class replacement for `mlx_lm`'s `KVCache` with integrated chunking
- `scripts/` — Handful scripts for tests, local servers, and EXO-clusters

## Key Design Decisions
- **mlx_lm Monkey-patch:** Seamlessly integrates directly into `make_prompt_cache` and `KVCache`, guaranteeing memory compression across modules globally.
- **Asymmetric Compression:** Keys are compressed via highly accurate `TurboQuant`, while Values are heavily shrunk via standard `PolarQuant`.
- **Heavy Hitter Caching / FP16 Sink:** First 128 context tokens stay completely uncompressed, saving instruction-following metrics at extreme bitrates.

## Clients
- `ios/` — **TurboMic**, the SwiftUI iPhone client. Audio capture and speech-to-text run on the phone; transcripts are analysed by `scripts/run_assistant_server.py` on a Mac over the local network. See [`ios_app.md`](ios_app.md) for the full breakdown and the path to on-device inference via MLX Swift.
