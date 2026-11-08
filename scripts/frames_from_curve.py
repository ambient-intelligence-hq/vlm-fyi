"""Turn frame-curve runs (probe/frame_curve.py) into data/frames.json for the export.

For every route the curve run measured, take the accepted preprocessing setting
(reference or provider-specific), apply it to the 5-minute reference clip
(1920x1080, 30 fps, 308 s), and record frames, frame groups, frame size and tokens.
As an independent check, the same setting predicts the video tokens per minute the
density probe measured earlier on that clip (real video, not a counter clip).

Routes whose repeated clips billed differently are recorded as "varies"; routes no
setting explains as "not_determined" (billed tokens only). No model output is used.

    uv run --no-project --with numpy,tokenizers python scripts/frames_from_curve.py \
        probe/results/curve-A/curve.json probe/results/curve-B/curve.json ...
"""
from __future__ import annotations

import re
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe"))
import frame_curve as F  # noqa: E402

DATA = ROOT / "data"
REF = {"width": 1920, "height": 1080, "fps": 30, "frames": 9240, "seconds": 308}
SHORT = {"width": 1920, "height": 1080, "fps": 30, "frames": 900, "seconds": 30}
# Variants the density probe used when a provider refused the original clip.
CLIP_DESC = {"baker_5min": "the 5-minute 1080p reference clip", "baker_5min_h264": "the 5-minute 1080p reference clip (H.264 copy)",
             "baker_5min_360p": "a 360p copy of the 5-minute reference clip", "baker_4min_h264": "the first 4 minutes of the reference clip",
             "baker_1min_h264": "the first minute of the reference clip"}
VARIANTS = {"baker_5min": REF, "baker_5min_h264": REF,
            "baker_5min_360p": {**REF, "width": 640, "height": 360},
            "baker_4min_h264": {**REF, "frames": 7200, "seconds": 240},
            "baker_1min_h264": {**REF, "frames": 1800, "seconds": 60}}
MERGE = {"kimi": 4, "glm": 2, "qwen": 2, "nemotron": 2, "muse": 2, "gemma": 1, "gemini": 1}
UNIT = {"kimi": "4-frame chunk", "glm": "2-frame group", "qwen": "2-frame group", "nemotron": "2-frame tubelet",
        "muse": "2-frame clip", "gemma": "frame", "gemini": "frame"}
REFERENCE = {
    "kimi": "Kimi K3.x reference: 8 fps, 4-frame chunks, up to 16,384 patches per frame",
    "glm": "GLM-5.3 reference: 2 fps, 2-frame groups, 240,000-token budget per video",
    "qwen": "Qwen3-VL reference: 2 fps (4 to 768 frames), 2-frame groups, 25.2-megapixel budget per video, timestamp per group",
    "gemma": "Gemma 4 reference: 32 frames per video, 70 soft tokens each",
    "nemotron": "Nemotron 3.5 VL reference (vLLM): 2-frame tubelets, ~1,024 patches per frame, EVS pruning",
    "muse": "Muse Spark reference (vLLM): 2 fps up to 600 frames, 2-frame clips, at most 256 tokens per frame",
    "gemini": "Google's documented rate: 1 frame per second at the default media resolution",
}


def entry_id(target: str) -> str:
    model, route = target.split(" · ", 1)
    if route.startswith("openrouter/"):
        return f"openrouter-{route[len('openrouter/'):].replace('/', '-')}/{model}"
    return f"{route}/{model}"


def describe(fam: str, knobs: dict) -> str:
    """The provider's setting in words (only what differs from the reference)."""
    k = {**knobs}
    parts = []
    if fam == "glm":
        if k.get("fixed_frames"):
            parts.append(f"{k['fixed_frames']} frames per video (fewer for very short clips)")
        if k.get("budget") and k["budget"] != 240000:
            parts.append(f"a budget of about {k['budget']:,} tokens per video instead of 240,000")
    elif fam == "qwen":
        if k.get("fixed_frames"):
            parts.append(f"{k['fixed_frames']} frames per video (fewer for very short clips)")
        if not k.get("capped") and k.get("budget") and k["budget"] < 8_000_000:
            parts.append(f"a pixel budget of about {k['budget'] / 1e6:.1f} megapixels per video instead of 25.2, so frames are very small")
        if k.get("capped"):
            parts.append(f"frames capped at about {k.get('max_video_tokens', 768)} tokens per group, no total budget"
                         if k.get("budget", 0) > 10 ** 9 else f"frames capped at {k.get('max_video_tokens')} tokens per group")
        if k.get("max_frames", 768) != 768:
            parts.append(f"at most {k['max_frames']} frames per video")
        if k.get("min_frames", 4) != 4:
            parts.append(f"at least {k['min_frames']} frames")
    elif fam == "nemotron":
        if k.get("num_frames", -1) > 0:
            parts.append(f"at most {k['num_frames']} frames per video" if k.get("fps", -1) > 0 else f"{k['num_frames']} frames per video")
        if k.get("fps", -1) > 0:
            parts.append(f"{k['fps']:g} fps")
        q = k.get("q", 0.7)
        parts.append("no token pruning" if q == 0 else f"EVS keeps {round((1 - q) * 100)}% of video tokens")
    elif fam == "muse":
        if k.get("max_frames", 600) != 600:
            parts.append(f"at most {k['max_frames']} frames per video")
        if k.get("max_tok_frame", 256) != 256:
            parts.append(f"{k['max_tok_frame']} tokens per frame instead of 256")
    elif fam == "gemma":
        if k.get("clamp"):
            parts.append("all frames when the video has fewer than 32")
    elif fam == "kimi":
        if k.get("video_budget"):
            parts.append("the 655,360-patch video limit applied across the whole video, so frames shrink once a video passes about 4 seconds")
        if k.get("max_tokens"):
            parts.append(f"each video limited to about {round(k['max_tokens'], -3):,} visual tokens, which is 15 chunks (60 frames) at 1080p")
    return "; ".join(parts)


def main(paths: list[str]) -> None:
    caps = {e["id"]: e for e in json.loads((DATA / "capabilities.json").read_text())["entries"]}
    routes: dict[str, dict] = {}
    for p in paths:                                    # later runs add to (and override clips of) earlier ones
        m = re.search(r"curve-(\d{4})(\d{2})(\d{2})", str(p))
        run_date = f"{m[1]}-{m[2]}-{m[3]}" if m else None
        for r in json.loads(Path(p).read_text()):
            for c in r.get("clips", []):
                c.setdefault("date", run_date)
            old = routes.get(r["target"])
            if old and not old.get("error") and not r.get("error"):
                clips = {c["clip"]: c for c in old["clips"]}
                clips.update({c["clip"]: c for c in r["clips"] if c.get("billed") is not None or c["clip"] not in clips})
                routes[r["target"]] = {**old, "clips": list(clips.values()), "repeats": old.get("repeats", []) + r.get("repeats", []),
                                       "run": old["run"] + "+" + Path(p).parent.name}
            elif not r.get("error") or not old:
                routes[r["target"]] = {**r, "run": Path(p).parent.name}
    out: dict[str, dict] = {}
    for target, r in sorted(routes.items()):
        eid = entry_id(target)
        if eid not in caps:
            print(f"skip {target}: no published entry {eid}")
            continue
        fam = r["family"]
        bills = [{"clip": c["clip"], "billed": c.get("billed"), **({"error": c.get("kind")} if c.get("billed") is None else {})}
                 for c in r.get("clips", [])]
        rec: dict = {"target": target, "run": r["run"], "family": fam, "unit": UNIT[fam], "bills": bills,
                     "repeats": r.get("repeats", [])}
        if r.get("error"):
            rec.update(status="error", note=r["error"][:200])
            out[eid] = rec
            continue
        a = F.analyse(r)
        rec["verdict"] = a["verdict"]
        if a.get("n_clips", 0) < 4:
            kinds = sorted({c.get('kind') for c in r.get('clips', []) if c.get('billed') is None})
            first = next((c for c in r.get('clips', []) if c.get('billed') is None), {})
            rec.update(status="no_data", kinds=kinds, evidence=(first.get('message') or '')[:240],
                       note=f"no clip was billed ({', '.join(map(str, kinds))})")
        elif "openrouter/auto" in target:
            rec.update(status="varies", note="OpenRouter picks the upstream for each request, so what reaches the model "
                       "depends on which provider serves it. Pin an upstream for a fixed result.")
        elif a.get("repeat_identical") is False:
            rec.update(status="varies", note="Identical requests billed different amounts: " + ", ".join(
                f"{x['clip'].replace('counter_', '').replace('.mp4', '')} {x['first']:,} then {x['billed']:,}" for x in r["repeats"]
                if x.get("billed") is not None and x["billed"] != x["first"]))
        elif not a.get("accepted"):
            best = a.get("closest") or {}
            rec.update(status="not_determined", closest=best.get("knobs"), closest_residual=best.get("max_residual"),
                       note="Billing is consistent, but no preprocessing setting reproduces it exactly, so only tokens are published.")
            if best.get("max_residual") is not None and best["max_residual"] <= 2:
                cf = F.predict(fam, best["knobs"], REF)
                rec["note"] = (f"Closest fit, within {best['max_residual']:g} tokens on every clip: {describe(fam, best['knobs']) or 'the reference'}, "
                               f"which gives {cf['frames']} frames per video, in {cf['units']} groups.")
        else:
            # Apply every accepted setting to the reference clip; publish only if they agree.
            # Frames on the clip the density probe used for this route (a shorter or smaller copy when the
            # provider refused the full one), so frames and tokens per minute describe the same video.
            tok0 = caps[eid]["limits"].get("video_tokens_per_min", {})
            name = ((tok0.get("video") or "baker_5min").split(",")[0]).strip()
            if tok0.get("status") == "measured":
                clip, clip_desc = VARIANTS.get(name, REF), CLIP_DESC.get(name, CLIP_DESC["baker_5min"])
            else:
                # The reference clip was refused (too large or too long): use the longest 1080p clip this route billed.
                ok = [c for c in r["clips"] if c.get("billed") is not None and c["width"] == 1920 and c["fps"] == 30]
                big = max(ok, key=lambda c: c["seconds"])
                clip = {"width": 1920, "height": 1080, "fps": 30, "frames": big["frames"], "seconds": big["seconds"]}
                clip_desc = f"a {big['seconds']:g}-second 1080p clip, the longest this route accepted"
                rec["clip_tokens_per_min"] = round(big["billed"] / (big["seconds"] / 60))
                rec["clip_seconds"] = big["seconds"]
                # A clip at least as long as the reference refused outright (too long or too large) is a
                # verdict on the reference clip too.
                no = [c for c in r["clips"] if c.get("billed") is None and c["seconds"] >= REF["seconds"]
                      and c.get("kind") in ("video_duration", "media_bytes", "decoded_bytes", "request_bytes", "download_bytes")]
                if no:
                    c = min(no, key=lambda c: c["seconds"])
                    rec["refused_long"] = {"seconds": c["seconds"], "kind": c["kind"], "message": c.get("message", ""), "date": c.get("date")}
            preds = []
            for h in a["accepted_all"]:
                p = F.predict(fam, h["knobs"], clip)
                s = F.predict(fam, h["knobs"], SHORT)
                preds.append((h, p, s))
            # One more exact bill: the density probe's tokens on the real reference clip. Settings that
            # mispredict it (by more than 0.1%) are ruled out.
            tm = tok0.get("value") if tok0.get("status") == "measured" else None
            if tm and len({p["frames"] for _, p, _ in preds}) > 1:
                def billed_per_min(h, p):
                    return (p["visual"] + h["k"] * p["units"] + h["d"] * p["digits"] + h["a"]) / (clip["seconds"] / 60)
                kept = [x for x in preds if abs(billed_per_min(x[0], x[1]) - tm) <= 0.001 * tm]
                if kept:
                    rec["ruled_out_by_reference_clip"] = len(preds) - len(kept)
                    preds = kept
            frames = {p["frames"] for _, p, _ in preds}
            preds.sort(key=lambda x: x[1]["frames"])          # if settings disagree, publish the lowest as a floor
            h, p, s = preds[0]
            documented = None
            if fam == "gemini" and len(frames) > 1:
                # Billing fixes tokens per second but not how they split into frames (one 66-token frame per
                # second bills the same as two 33-token frames). Use Google's documented rate.
                h, p, s = next(x for x in preds if not x[0]["knobs"])
                frames = {p["frames"]}
                documented = ("Billing shows 66 tokens per second of video. The frame count follows Google's documented "
                              "one frame per second: billing alone can't tell one 66-token frame from two 33-token frames.")
            ref_match = a["verdict"] == "matches reference"
            minutes = clip["seconds"] / 60
            rec.update(status="measured" if len(frames) == 1 else "at_least", reference=ref_match,
                       setting=REFERENCE[fam] if ref_match else describe(fam, h["knobs"]),
                       knobs=h["knobs"], fit={"k": h["k"], "a": h["a"], "d": h["d"]},
                       frames_per_video=p["frames"], units_per_video=p["units"], tokens_per_unit=p["per_unit"],
                       frame_size=p.get("size"), video_minutes=round(minutes, 2), clip=clip_desc,
                       value=round(p["frames"] / minutes, 1), frames_per_min_short_clip=round(s["frames"] / (SHORT["seconds"] / 60), 1),
                       hypotheses=len(preds))
            rec["capped"] = rec["value"] < 0.6 * rec["frames_per_min_short_clip"]
            if documented:
                rec["note"] = documented
            ref_pred = next((x for x in preds if not x[0]["knobs"]), None)
            if ref_pred and len(frames) == 1:
                h, p, s = ref_pred                     # the reference fits: report its values
            sizes = {x[1].get("size") for x in preds if x[1]["frames"] == p["frames"]}
            if len(sizes) > 1 and not ref_pred:
                rec["frame_size"] = None
                rec["size_range"] = sorted(sizes)
            if len(frames) > 1:
                rec["frames_range"] = [min(frames), max(frames)]
                rec["note"] = (f"At least {min(frames)} frames per video: clips longer than the ones this provider accepted "
                               "would be needed to find its cap.")
            # Independent check against the density probe's measured tokens/min on the real clip.
            tok = caps[eid]["limits"].get("video_tokens_per_min", {})
            if tok.get("status") == "measured" and tok.get("value"):
                var = VARIANTS.get((tok.get("video") or "baker_5min").split(",")[0].strip(), REF)
                q = F.predict(fam, h["knobs"], var)
                billed = q["visual"] + h["k"] * q["units"] + h["d"] * q["digits"] + h["a"]
                pred = billed / (var["seconds"] / 60)
                rec["check"] = {"clip": tok.get("video"), "predicted_tokens_per_min": round(pred),
                                "measured_tokens_per_min": tok["value"],
                                "diff_pct": round(100 * (tok["value"] - pred) / pred, 2)}
        out[eid] = rec
        c = rec.get("check")
        print(f"{rec['status']:15} {eid:58} {str(rec.get('value', '')):>6}/min "
              f"{str(rec.get('frames_per_video', '')):>5}f {str(rec.get('frame_size') or ''):>9} "
              + (f"check {c['predicted_tokens_per_min']:>7} vs {c['measured_tokens_per_min']:>7} ({c['diff_pct']:+.2f}%)" if c else ""))
    (DATA / "frames.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"→ {DATA / 'frames.json'} ({len(out)} routes)")


if __name__ == "__main__":
    main(sys.argv[1:])
