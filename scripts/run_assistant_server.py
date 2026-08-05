#!/usr/bin/env python3
"""
TurboQuant assistant server — the Mac-side backend for the TurboMic iPhone app.

`run_server.py` gives you the plain OpenAI-compatible surface from `mlx_lm`.
This one adds the two endpoints the phone actually wants:

    GET  /health      -> which model is loaded and whether it is ready
    POST /v1/analyze  -> transcript in, structured JSON out

The prompt lives here rather than on the phone so it can be tuned without
shipping a new build. The KV cache is compressed by TurboQuant exactly as in
`run_server.py`, which is what makes long realtime sessions affordable: the
running summary plus a growing transcript is a long context, and that context is
almost entirely KV cache.

Usage:
    python scripts/run_assistant_server.py \\
        --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \\
        --host 0.0.0.0 --port 8080
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SYSTEM_PROMPT = """You are a note-taking assistant. You receive a raw, imperfect \
speech-to-text transcript and pull out only what matters. Ignore filler, small \
talk and transcription noise.

Answer with a single JSON object and nothing else. No prose, no markdown fences.

Schema:
{
  "summary": "two or three sentences",
  "items": [
    {
      "kind": "task|decision|fact|question|date|contact|idea",
      "text": "one short sentence",
      "who": "person responsible, or null",
      "due": "deadline as stated, or null",
      "confidence": 0.0
    }
  ]
}

Rules:
- Write summary and text in the same language as the transcript.
- Never invent anything that is not in the transcript. When the transcript is too \
short or says nothing of substance, return an empty items array.
- confidence is your own 0.0-1.0 estimate that the item is real and correctly read.
- At most 12 items."""

MAX_TRANSCRIPT_CHARS = 24000

# Loaded once at startup, then shared by every request thread.
_MODEL = None
_TOKENIZER = None
# mlx generation is not thread-safe; serialize it.
_GENERATE_LOCK = threading.Lock()


def build_user_prompt(payload: dict) -> str:
    """Assemble the user turn from what the phone sent."""
    parts = []

    previous = (payload.get("previous_summary") or "").strip()
    if previous:
        parts.append("Summary of the conversation so far:\n" + previous)

    known = [str(k).strip() for k in (payload.get("known_insights") or []) if str(k).strip()]
    if known:
        listed = "\n".join("- " + k for k in known[:30])
        parts.append("Already captured — do not repeat these:\n" + listed)

    if payload.get("mode") == "realtime":
        parts.append(
            "This is a live fragment of an ongoing recording. Extract only what is new in it."
        )

    transcript = (payload.get("transcript") or "").strip()
    # Keep the tail: in a rolling session the recent speech is the part being analysed.
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[-MAX_TRANSCRIPT_CHARS:]
    parts.append("Transcript:\n" + transcript)

    return "\n\n".join(parts)


def apply_chat_template(tokenizer, system: str, user: str) -> str:
    """Render the chat turns, falling back to a plain prompt on older tokenizers."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{system}\n\n{user}\n\nJSON:"


def generate_text(prompt: str, max_tokens: int) -> str:
    """Single-flight generation against the loaded model."""
    from mlx_lm import generate

    kwargs = {"max_tokens": max_tokens, "verbose": False}

    # `temp=` was replaced by a sampler object in newer mlx-lm; use whichever exists.
    try:
        from mlx_lm.sample_utils import make_sampler

        kwargs["sampler"] = make_sampler(temp=0.2)
    except Exception:
        pass

    with _GENERATE_LOCK:
        return generate(_MODEL, _TOKENIZER, prompt=prompt, **kwargs)


def extract_json(raw: str) -> dict:
    """
    Pull the outermost JSON object out of a model answer.

    Small quantized models still wrap JSON in prose or fences now and then, and a
    dropped analysis pass is worse than a lenient parser.
    """
    text = raw.strip()
    if "```" in text:
        blocks = [b for b in text.replace("```json", "```").split("```") if "{" in b]
        if blocks:
            text = blocks[0]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("no JSON object in model output")

    return json.loads(text[start : end + 1])


def normalize(parsed: dict) -> dict:
    """Coerce whatever the model produced into the schema the app decodes."""
    valid_kinds = {"task", "decision", "fact", "question", "date", "contact", "idea"}
    items = []

    for item in (parsed.get("items") or [])[:12]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        kind = str(item.get("kind") or "fact").strip().lower()
        if kind not in valid_kinds:
            kind = "fact"

        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        def optional(key):
            value = item.get(key)
            if value is None:
                return None
            value = str(value).strip()
            return value or None

        items.append(
            {
                "kind": kind,
                "text": text,
                "who": optional("who"),
                "due": optional("due"),
                "confidence": min(max(confidence, 0.0), 1.0),
            }
        )

    return {"summary": str(parsed.get("summary") or "").strip(), "items": items}


class AssistantHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TurboQuantAssistant/0.1"

    model_name = "unknown"

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send(
                200,
                {
                    "status": "ok" if _MODEL is not None else "loading",
                    "model": self.model_name,
                    "compression": "turboquant k8/v3",
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/analyze":
            self._send(404, {"error": "not found"})
            return

        if _MODEL is None:
            self._send(503, {"error": "model still loading"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"bad request body: {exc}"})
            return

        transcript = (payload.get("transcript") or "").strip()
        if len(transcript) < 10:
            # Too little to say anything honest about.
            self._send(200, {"summary": "", "items": []})
            return

        prompt = apply_chat_template(
            _TOKENIZER, SYSTEM_PROMPT, build_user_prompt(payload)
        )

        try:
            raw = generate_text(prompt, max_tokens=int(payload.get("max_tokens", 900)))
        except Exception as exc:
            self._send(500, {"error": f"generation failed: {exc}"})
            return

        try:
            result = normalize(extract_json(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(502, {"error": f"model returned unparseable output: {exc}"})
            return

        self._send(200, result)

    def log_message(self, fmt, *args):
        sys.stderr.write("[assistant] %s\n" % (fmt % args))


def main():
    parser = argparse.ArgumentParser(description="TurboQuant assistant server")
    parser.add_argument(
        "--model",
        default="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        help="mlx-lm model id or local path",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--k-bits", type=int, default=8, help="TurboQuant key angle bits")
    parser.add_argument("--v-bits", type=int, default=3, help="TurboQuant value angle bits")
    parser.add_argument(
        "--sink-size", type=int, default=128, help="leading tokens kept uncompressed"
    )
    args = parser.parse_args()

    global _MODEL, _TOKENIZER

    # The cache patch has to land before any mlx_lm layer is constructed.
    from turboquant_mlx.plugins.cache_plugin import apply_turboquant_cache

    apply_turboquant_cache(
        k_theta_bits=args.k_bits,
        v_theta_bits=args.v_bits,
        fp16_sink_size=args.sink_size,
    )

    try:
        from mlx_lm import load
    except ImportError:
        print("mlx-lm is not installed. Run: pip install -e .", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.model} ...")
    _MODEL, _TOKENIZER = load(args.model)
    AssistantHandler.model_name = args.model

    server = ThreadingHTTPServer((args.host, args.port), AssistantHandler)
    print(f"Assistant server ready on http://{args.host}:{args.port}")
    print("  GET  /health")
    print("  POST /v1/analyze")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
