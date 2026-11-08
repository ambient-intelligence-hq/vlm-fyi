"""Frames the model can read back from a 5-minute counter clip (every frame shows its number).

This is the one video measure that depends on the model's reply, so it is run three
times per route and reported as a median and range, next to (not instead of) the
billing-derived tokens and frames. Each run also records the tokens billed for the
clip, which must agree with the billing curve.

    uv run --no-project --with aiohttp,pillow,boto3,rich,python-dotenv \
        python read_long.py --targets targets_curve.toml --targets targets_new.toml --only "kimi-k3.1-pro · fireworks" ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import dotenv_values, load_dotenv

import probe as P

P.CALIBRATION.update({
    "long": {"name": "counter_1080p_300s", "s3_key": "probe/ref/counter_1080p_300s.mp4", "w": 1920, "h": 1080, "fps": 30,
             "seconds": 300, "font": 520},
    "long_360p": {"name": "counter_360p_300s", "s3_key": "probe/ref/counter_360p_300s.mp4", "w": 640, "h": 360, "fps": 30,
                  "seconds": 300, "font": 180},
    "minute": {"name": "counter_1080p_60s", "s3_key": "probe/ref/counter_1080p_60s.mp4", "w": 1920, "h": 1080, "fps": 30,
               "seconds": 60, "font": 520},
})
REFUSED = {"decoded_bytes", "media_bytes", "request_bytes", "download_bytes", "invalid_media", "video_duration", "image_bytes"}
MINUTE_ONLY = ("kimi-k3.1-pro · fireworks",)   # a 5-minute video overflows the 1M context there
INLINE = ("openrouter/z-ai", "openrouter/cloudflare", "moonshot")


async def route(http, t: P.Target, fx: P.Fixtures, runs: int, sem) -> dict:
    which = "minute" if t.name in MINUTE_ONLY else "long"
    if any(x in t.name for x in INLINE):
        t.video_inline = True
    res = {"target": t.name, "clip": P.CALIBRATION[which]["name"], "runs": []}
    async with sem:
        for i in range(runs):
            P.CTX.set({"dimension": "read_long", "value": f"run {i + 1}"})
            r = await P.frame_calibration(http, t, fx, which, None)
            if not r.get("ok") and r.get("kind") in REFUSED and which == "long":
                which = "long_360p"
                res["clip"] = P.CALIBRATION[which]["name"]
                res["fallback"] = f"1080p clip refused ({r.get('kind')}): {str(r.get('message', ''))[:160]}"
                r = await P.frame_calibration(http, t, fx, which, None)
            res["runs"].append({k: r.get(k) for k in ("ok", "kind", "message", "error", "retry", "frames_read", "visual_tokens",
                                                     "first", "last", "median_gap_frames", "latency_s", "request_id", "reply")})
            P.console.print(f"[dim]{datetime.now():%H:%M:%S}[/] {t.name:52} {res['clip']:20} run {i + 1}: "
                            f"read {r.get('frames_read', '✗ ' + str(r.get('kind') or r.get('error')))}  "
                            f"tokens {r.get('visual_tokens')}", highlight=False)
            if not r.get("ok") and r.get("kind") in ("no_video", "no_route", "auth_or_billing", "credit_guard"):
                break
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", action="append", required=True)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    load_dotenv(P.HERE / ".env", override=False)
    secrets = {k: v for k, v in dotenv_values(P.HERE / ".env.local").items() if v}
    out_dir = P.RESULTS / f"readlong-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    P.LOG = P.RequestLog(out_dir / "requests.jsonl")
    P.DENSITY_TIMEOUT = 900

    async def run_all():
        fx = P.Fixtures()
        for which in ("long", "long_360p", "minute"):        # build and upload the counter clips once
            await asyncio.to_thread(fx.counter_url, which)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as http:
            targets = await P.load_targets(http, [Path(p) for p in args.targets], secrets, args.only, args.exclude)
            P.GUARD = P.CreditGuard(http, secrets.get("OPENROUTER_API_KEY"), 2.0)
            P.console.print(f"read_long · {len(targets)} routes · OpenRouter balance {await P.GUARD.remaining()}\n[dim]→ {out_dir}[/]")
            lanes: dict[str, asyncio.Semaphore] = {}
            sems = [lanes.setdefault(f"{t.service}:{t.pin}" if t.service == "openrouter" else t.service, asyncio.Semaphore(2))
                    for t in targets]
            res = await asyncio.gather(*(route(http, t, fx, args.runs, s) for t, s in zip(targets, sems)))
            P.console.print(f"[dim]OpenRouter balance now {await P.GUARD.remaining()}[/]")
            return res

    results = asyncio.run(run_all())
    (out_dir / "read_long.json").write_text(json.dumps(results, indent=1))
    for r in results:
        reads = [x.get("frames_read") for x in r["runs"] if x.get("frames_read")]
        toks = sorted({x.get("visual_tokens") for x in r["runs"] if x.get("visual_tokens")})
        print(f"{r['target']:52} {r['clip']:20} read {sorted(reads)}  tokens {toks}")
    print(f"→ {out_dir / 'read_long.json'}")


if __name__ == "__main__":
    main()
