# TurboQuant-MLX Development Plan

## Phase 1: Mathematical Foundation (Pure Python / NumPy)
- [x] Unbiased rounding logic
- [x] QJL (Quantized Johnson-Lindenstrauss) concept
- [x] PolarQuant concept
- [x] Base accuracy tests via simulated Attention output

## Phase 2: Implementation (Apple MLX Framework)
- [x] Core structures ported to `mlx.core`
- [x] Asymmetric integration via Monkey-patching `mlx-lm`
- [x] Memory tests checking strict allocation maps

## Phase 3: Hardware Verification
- [x] Performance benchmarking (NumPy vs Metal execution time)

## Phase 4: Productionization & Deployment
- [x] Structured package (`pyproject.toml`)
- [x] Attention sinks (FP16 uncompressed pre-fills)
- [x] `run_server.py` implementation mapping OpenAI endpoints
- [x] EXO integration scripts for decentralized Mac networks

## Phase 5: Community Release
- [x] Internationalization (English format globally applied)
- [ ] Receive pull requests and model-specific tuning 

## Phase 6: TurboMic (iPhone client)
- [x] `run_assistant_server.py` — structured `/v1/analyze` endpoint over the patched cache
- [x] SwiftUI app scaffold: single-button UI, on-device transcription, session storage
- [x] Two capture modes (button / realtime) behind one analysis pipeline
- [ ] First device build: signing, on-device recognition check
- [ ] `OnDeviceAnalyzer` via MLX Swift (Llama 3.2 1B / Qwen 2.5 1.5B)
- [ ] Port `PolarQuantCompressor` to MLX Swift and validate against the Python reference
- [ ] Migrate to `SpeechAnalyzer` (iOS 26) to drop recognition-task rotation
