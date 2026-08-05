# TurboMic

iPhone client for TurboQuant: one big button, on-device transcription, and a
local model that keeps only what matters.

Full design notes, iOS constraints and the on-device roadmap live in
[`../docs/ios_app.md`](../docs/ios_app.md).

## Build

```bash
brew install xcodegen
xcodegen generate
open TurboMic.xcodeproj
```

Set a signing team on the `TurboMic` target and run on a real device.

## Backend

```bash
# on your Mac, from the repo root
python scripts/run_assistant_server.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
```

Enter `http://<mac-name>.local:8080` in the app's Settings and tap **Test
connection**. Any OpenAI-compatible endpoint also works — the app falls back to
`/v1/chat/completions` when `/v1/analyze` is absent.

## Layout

```
Sources/
  Models/      CaptureMode, Insight, Session
  Audio/       AudioCaptureService (AVAudioEngine), SpeechTranscriptionService (Speech)
  Analysis/    Analyzer protocol + prompt, RemoteAnalyzer, InsightPipeline
  State/       AppSettings, SessionStore, RecorderViewModel
  Views/       ContentView, RecordButton, InsightListView, SessionListView, SettingsView
Resources/
  Info.plist   mic + speech purpose strings, background audio mode
```

To add on-device inference, implement `Analyzer` once more and return it from
`AppSettings.makeAnalyzer()`. Nothing else moves.

## Note

Audio never leaves the phone; speech recognition is on-device. Transcript text is
sent only to the backend address you configure. Audio retention is off by default.
Recording other people is regulated in many places — check what applies where you
are.
