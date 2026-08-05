# TurboMic — iPhone app on top of TurboQuant

## 1. What this repo is today

Before designing anything, it is worth being blunt about the starting point.
`turboquant-mlx` is **not** an agent. It is a KV-cache compression library:

| Component | What it does |
|---|---|
| `src/turboquant_mlx/polarquant.py` | Cartesian→polar quantization of key/value vectors |
| `src/turboquant_mlx/turboquant.py` | PolarQuant + QJL residual correction |
| `src/turboquant_mlx/plugins/cache_plugin.py` | Monkey-patches `mlx_lm`'s `KVCache` with the compressed one |
| `scripts/run_server.py` | Starts `mlx_lm`'s OpenAI-compatible server with the patch applied |

There is no audio capture, no transcription and no extraction logic anywhere in
the tree. So "port the project to iPhone" is not a port — it is a new app that
**uses** this project. And it cannot be a literal port anyway: Python + the MLX
Python bindings do not run on iOS. Only MLX Swift does.

That leaves two real ways for this repo to matter to an iPhone app, and they are
sequential, not exclusive:

* **Now — the Mac is the brain.** The phone captures and transcribes; a Mac on
  the same network runs the model with TurboQuant compression. Zero porting.
* **Later — the phone is the brain.** PolarQuant is reimplemented in MLX Swift so
  a small model can hold a long context inside an iPhone's memory budget. This is
  where the compression stops being a nice-to-have (see §6).

## 2. Architecture

```
┌─────────────────────── iPhone (ios/) ────────────────────────┐
│  RecordButton  ──tap──►  RecorderViewModel                   │
│                             │                                │
│              AudioCaptureService  (AVAudioEngine, background)│
│                             │ PCM buffers                    │
│              SpeechTranscriptionService (Speech, on-device)  │
│                             │ committed segments             │
│                    InsightPipeline  (when to analyse)        │
│                             │ AnalysisRequest                │
│                      RemoteAnalyzer  ───HTTP───┐             │
│                             │                  │             │
│                    SessionStore (JSON on disk) │             │
└────────────────────────────────────────────────┼─────────────┘
                                                 │ LAN / Tailscale
┌────────────────────────── Mac ─────────────────▼─────────────┐
│  scripts/run_assistant_server.py                             │
│      POST /v1/analyze  → prompt → mlx_lm.generate            │
│                                      │                       │
│      apply_turboquant_cache(k=8, v=3, sink=128)              │
│                                      │                       │
│                             compressed KV cache              │
└──────────────────────────────────────────────────────────────┘
```

Audio never leaves the phone. Speech-to-text is `SFSpeechRecognizer` with
`requiresOnDeviceRecognition = true`. Only the resulting text crosses the network,
to an address the user typed in themselves.

## 3. The two modes

Both are the same pipeline with a different trigger policy — that is the whole
reason `InsightPipeline` is a separate object from `RecorderViewModel`.

**Button mode** (`CaptureMode.button`). Tap → record → tap → stop. One analysis
pass over the full transcript at the end. Cheap, predictable, and the right
default: one request per recording.

**Realtime mode** (`CaptureMode.realtime`). Continuous listening. Committed
speech segments accumulate and a pass fires when either:

* ≥ 220 new characters have arrived and ≥ 12 s passed since the last pass, or
* ≥ 45 s passed and there are at least 60 new characters.

One request is in flight at a time. A slow model backs the queue up rather than
stacking concurrent requests on it. Each pass is given the running summary and
the list of already-extracted items and told not to repeat them; anything that
slips through is caught by `Insight.dedupeKey` on the client.

## 4. Things iOS will not let you skip

These constrain the product, so they are design inputs rather than footnotes.

**Background recording works, but visibly.** `UIBackgroundModes: [audio]` plus an
active `AVAudioSession` keeps `AVAudioEngine` running when the app is backgrounded
or the screen is locked. iOS shows the orange microphone dot the entire time and
the user can see it in Control Centre. There is no supported way to record
invisibly, and there should not be.

**Recognition tasks expire.** A single `SFSpeechRecognitionTask` is terminated by
the system after roughly a minute. `SpeechTranscriptionService` rotates the
request every 50 s — commits the final hypothesis, cancels, and immediately opens
a new one. Without this, "records everything" quietly becomes "records the first
minute".

**Battery.** Continuous mic + on-device ASR is roughly 8–12 %/hour on a recent
iPhone, before any local model. It is the reason button mode is the default.

**App Store review.** An always-listening recorder is reviewed strictly.
A clear in-app indicator, an explicit start action, and honest purpose strings
are the difference between approval and rejection. For personal use, sideloading
via a free developer account works and expires every 7 days.

**Consent.** Recording conversations is regulated differently in different places
— several US states and much of the EU require all-party consent, not just yours.
This affects what the app should encourage, not just its legal copy. It is why
audio retention is off by default and why the recording indicator is loud.

## 5. Building it

```bash
brew install xcodegen
cd ios
xcodegen generate          # produces TurboMic.xcodeproj from project.yml
open TurboMic.xcodeproj
```

Set your signing team on the `TurboMic` target and run on a device — the
simulator has no usable microphone path for this and on-device recognition
behaves differently there.

On the Mac:

```bash
pip install -e .
python scripts/run_assistant_server.py \
    --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --host 0.0.0.0 --port 8080
```

Then put `http://<mac-name>.local:8080` into the app's Settings and hit
**Test connection**. Note that `--host 0.0.0.0` exposes the endpoint to your whole
local network with no authentication; on an untrusted network bind to `127.0.0.1`
and reach it over Tailscale instead.

The app also falls back to `POST /v1/chat/completions`, so plain
`scripts/run_server.py` — or any OpenAI-compatible endpoint — works too, just with
the prompt built on the phone instead of the server.

## 6. Moving the brain onto the phone

The Mac backend is the pragmatic v1, but it means the assistant only works at
home. Making it standalone is a bounded amount of work:

**Step 1 — MLX Swift inference (no compression).** Add
[`mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples) and run
Llama 3.2 1B or Qwen 2.5 1.5B at 4-bit. Implement `Analyzer` a second time as
`OnDeviceAnalyzer` and switch on it in `AppSettings.makeAnalyzer()`. Nothing else
in the app changes — that is what the protocol is for.

**Step 2 — port PolarQuant to Swift.** This is where this repo's code earns its
place. An iPhone gives an app roughly 3 GB before the jetsam killer steps in. A
4-bit 1.5B model is ~1 GB of weights, leaving very little for KV cache — and a
realtime session that has been running for an hour is exactly a long-context
workload. At the measured 5.3× cache reduction, an hour-long session fits where it
otherwise would not.

The port is mechanical. `PolarQuantCompressor` is ~90 lines of `mx.*` calls that
map one-to-one onto MLX Swift's `MLXArray`:

| Python | Swift |
|---|---|
| `mx.linalg.qr(H, stream=mx.cpu)` | `MLX.Linalg.qr(h, stream: .cpu)` |
| `mx.arctan2(odd, even)` | `MLX.atan2(odd, even)` |
| `mx.stack([even, odd], axis: -1)` | `MLX.stacked([even, odd], axis: -1)` |
| `mx.clip`, `mx.round` | `MLX.clip`, `MLX.round` |

The rotation matrix `R` is seeded (`seed=42`), so a Swift implementation can be
validated directly against the Python one: compress the same vectors on both
sides and compare reconstruction error. The harder half is not the math, it is
the cache integration — `cache_plugin.py` works by monkey-patching `mlx_lm`, which
has no equivalent in Swift. There you subclass or wrap `KVCache` from
`mlx-swift-examples` explicitly.

**Step 3 — replace `SFSpeechRecognizer`.** iOS 26's `SpeechAnalyzer`/
`SpeechTranscriber` removes the task-rotation dance and handles long-form audio
natively. WhisperKit is the alternative if you need languages Apple does not
cover on-device.

## 7. Status

`ios/` is a complete, self-contained scaffold: audio, rotation-safe transcription,
both trigger modes, dedup, persistence and a settings screen. It has not been
compiled — it was written on Linux, where no Xcode exists — so expect to fix
signing and possibly a few API details on first build. The Python server is
tested for prompt assembly, JSON extraction and normalization.
