# turboquant_mlx API

The library itself exposes no network API — it is a cache patch applied in-process
(`apply_turboquant_cache`). Two scripts do serve HTTP:

* `scripts/run_server.py` — `mlx_lm`'s OpenAI-compatible server, patched. See the
  `mlx-lm` docs for its surface.
* `scripts/run_assistant_server.py` — the backend for the TurboMic iPhone app,
  documented below.

## Assistant server

```bash
python scripts/run_assistant_server.py --model <mlx-lm model id> --host 0.0.0.0 --port 8080
```

`--host 0.0.0.0` publishes the endpoint to the local network with no
authentication. On an untrusted network, bind `127.0.0.1` and tunnel instead.

### `GET /health`

```json
{ "status": "ok", "model": "mlx-community/Meta-Llama-3-8B-Instruct-4bit", "compression": "turboquant k8/v3" }
```

`status` is `loading` until the model is resident.

### `POST /v1/analyze`

Request:

| Field | Type | Notes |
|---|---|---|
| `transcript` | string | Required. Under 10 characters returns an empty result. Truncated to the trailing 24 000 characters. |
| `previous_summary` | string | Running summary, so realtime passes stay coherent. |
| `known_insights` | string[] | Already-surfaced items; the model is told not to repeat them. First 30 are used. |
| `mode` | `"button"` \| `"realtime"` | Realtime adds a "only what is new" instruction. |
| `max_tokens` | int | Default 900. |

Response:

```json
{
  "summary": "…",
  "items": [
    {
      "kind": "task|decision|fact|question|date|contact|idea",
      "text": "…",
      "who": null,
      "due": null,
      "confidence": 0.82
    }
  ]
}
```

Unknown `kind` values are folded into `fact`, `confidence` is clamped to 0…1, and
at most 12 items are returned. Status codes: `503` while the model loads, `502`
when the model's output is not parseable as JSON, `500` on generation failure.
