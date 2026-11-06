# vlm.fyi

Measured request limits for vision-language model APIs: how many images and video frames,
how many bytes and how big a video file each provider accepts in one request, and how much
of a video the model actually sees. Live at **[vlm.fyi](https://vlm.fyi)**.

The same model id behaves differently on every provider. One stops at 112 frames per request
while another takes 384. One feeds the model 15 frames of a 5-minute video, another samples
120 a minute, and a third accepts the video and shows the model nothing. All of these
requests can return HTTP 200.

## The data

| File | What |
|---|---|
| [`/v1/capabilities.json`](https://vlm.fyi/v1/capabilities.json) | Current dataset, one entry per model × provider pair |
| [`/v1/capabilities.csv`](https://vlm.fyi/v1/capabilities.csv) | The same, flattened |
| [`/v1/changes.json`](https://vlm.fyi/v1/changes.json) | What changed between measurements |
| [`data/snapshots/`](data/snapshots) | A dated copy of every measurement |

Data is CC BY 4.0. See [vlm.fyi/api](https://vlm.fyi/api) for the schema.

## Methodology, in short

Each probe sends a minimal request ("Reply with OK.", 16 output tokens) through the provider's
OpenAI-compatible chat completions API. The request carries synthetic media built to stress
**one limit at a time**, so limits don't mask each other:

| Limit | Media |
|---|---|
| Images per request | tiny 64×64 JPEGs: only the count grows |
| 768×432 frames | real-size frames, the common working size for video pipelines |
| Inline request body | tiny JPEGs padded with comment segments: bytes grow, pixels don't |
| Images by URL | the same padded images, hosted and sent as https links |
| Video file size | a 5 s clip padded with an MP4 `free` box |
| Video density | a real 5:08 reference clip sent as `video_url` |

Each limit is found by doubling until a request fails, then bisecting between the last pass
and the first failure. The provider's error message is kept as evidence. Video density comes
from the provider's own billing: prompt tokens with the video minus prompt tokens without it.
That is converted to frames per minute with a frame-counter clip, which the model reads back
frame by frame. A "≥" value means the provider took everything we sent. Limits can depend
on your account and tier.

The full write-up is at [vlm.fyi/methodology](https://vlm.fyi/methodology).

## Running the probe

```bash
cd probe
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # also needs ffmpeg on PATH
```

Put API keys in `probe/.env.local` (gitignored). Each target in a `targets*.toml` file names
the variable holding its key in `api_key_env`, e.g. `OPENROUTER_API_KEY=...`. The URL and
video probes host media in an S3-compatible bucket, so also set `S3_ENDPOINT`, `S3_BUCKET`,
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. The reference videos are not in this repo;
the video-density probe expects them at the keys in `REF_VIDEOS` in `probe.py`.

```bash
python probe.py --targets targets.toml                        # every target in a file
python probe.py --targets targets_sweep.toml --only qwen3.8   # one model, every OpenRouter upstream
python probe.py --skip video_density                          # leave out a dimension
python probe.py --merge                                       # combine runs into results/probe-merged/
```

Each run writes `results/probe-<timestamp>/`: `probe.json` holds the results and
`requests.jsonl` holds every request and response. That log contains request ids and
account identifiers, so keep it private. OpenRouter targets stop when the balance drops
below `--openrouter-floor` (default $2). Run `python probe.py -h` for all options.

## Publishing a run

```bash
python scripts/export.py --probe probe/results/probe-merged/probe.json \
  --catalog models.json --runs-dir probe/results
```

`--catalog` is OpenRouter's model list (`curl https://openrouter.ai/api/v1/models`). The export
writes `data/capabilities.json`, a dated snapshot and `data/changes.json`. It scrubs URLs,
request ids, account ids and key-like strings from the evidence first.

## The site

An [Astro](https://astro.build) static site, deployed to GitHub Pages on every push to `main`.

```bash
pnpm install
pnpm dev        # http://localhost:4321
pnpm build      # dist/
```

## Corrections

Found a limit that's wrong or out of date? [Open an issue](https://github.com/ambient-intelligence-hq/vlm-fyi/issues)
with the request you sent and the error you got. For anything else: hello@vlm.fyi.

Built by [Ambient](https://github.com/ambient-intelligence-hq/ambient), the open-source video
engine for AI agents.
