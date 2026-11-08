"""Billing-only video measurement, checked against each model's reference preprocessing.

No model output is read anywhere. For every route we send counter clips of known
length, frame rate and resolution, and record the prompt tokens the provider bills
for each (prompt with the clip minus the same prompt without it). Separately, the
model's reference preprocessing (its Hugging Face processor config plus the
vLLM / transformers video code, reimplemented here; no remote code is run) says
how many frame units each clip becomes and what each unit costs:

    billed(clip) = visual_tokens(clip) + k * units(clip) + d * digits(clip) + a

visual_tokens, units and digits (digit characters in the per-unit timestamp labels)
come from the preprocessing. k (per-unit label text: timestamps, begin/end markers)
and a (wrapper) are the only fitted constants; d is fixed per hypothesis (1 when the
tokenizer splits numbers into digits). A hypothesis is accepted only if it reproduces
every clip to within one token. The reference settings are tested first; if they
fail, the provider's possible settings are searched (frame rate, fixed frame count,
frame cap, token budget, pruning rate). Frame counts are reported as determined only
when every accepted hypothesis gives the same frame counts.

    uv run --no-project --with aiohttp,pillow,boto3,rich,python-dotenv,numpy \
        python frame_curve.py --targets targets_curve.toml            # measure
    ... python frame_curve.py --analyze results/curve-*/curve.json    # re-analyse, no requests
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

PROMPT = "Describe the scene in one sentence."

# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------

LADDER = [(1920, 1080, 30, s) for s in (1, 2, 4, 8, 16, 32, 64, 128, 192, 360)] + \
         [(640, 360, 30, 8), (1280, 720, 30, 16), (1920, 1080, 24, 5)]
REPEAT = [(1920, 1080, 30, 2), (1920, 1080, 30, 64)]


def clip_name(w: int, h: int, fps: int, secs: float) -> str:
    return f"counter_{w}x{h}_{fps}fps_{secs:g}s.mp4"


def clip_frames(fps: int, secs: float) -> int:
    return max(1, round(fps * secs))


def _digits(labels: Iterable[str]) -> int:
    return sum(ch.isdigit() for s in labels for ch in s)


# Per-unit label text (timestamps) is counted with the model's own tokenizer (tokenizer.json, a data
# file read by the `tokenizers` library); without one, digit characters are the fallback.
TOKENIZER_DIR = Path(os.environ.get("FRAME_CURVE_TOKENIZERS", Path(__file__).resolve().parent / "tokenizers"))
_TOK: dict[str, Any] = {}


def text_tokens(fam: str, labels: tuple) -> int:
    if fam not in _TOK:
        f = TOKENIZER_DIR / f"{fam}.tokenizer.json"
        try:
            from tokenizers import Tokenizer
            _TOK[fam] = Tokenizer.from_file(str(f)) if f.exists() else None
        except ImportError:
            _TOK[fam] = None
    tok = _TOK[fam]
    if tok is None:
        return _digits(labels)
    return sum(len(e.ids) for e in tok.encode_batch(list(labels), add_special_tokens=False))


@lru_cache(maxsize=None)
def _linspace(last: int, num: int, mode: str = "round") -> tuple:
    a = np.linspace(0, last, num)
    return tuple((a.round() if mode == "round" else np.floor(a)).astype(int).tolist())


# --------------------------------------------------------------------------
# Reference preprocessing (reimplemented from the published code)
# --------------------------------------------------------------------------

# Kimi K3.x: moonshotai/Kimi-K3.1-Pro preprocessor_config + media_utils.navit_resize_image;
# vLLM kimi_k25_vision_fused.navit_resize_video; SGLang kimi_k25 (8 fps, 4-frame chunks).
def kimi_resize(width, height, patch=14, merge=2, in_patch_limit=16384, limit_one_side=512, min_patches=0):
    pw, ph = max(1.0, width // patch), max(1.0, height // patch)
    caps = (math.sqrt(in_patch_limit / (pw * ph)), limit_one_side * patch / width, limit_one_side * patch / height)
    scale = min(1.0, *caps)
    if min_patches and pw * ph * scale * scale < min_patches:      # hypothetical provider upscaling of small frames
        scale = min(math.sqrt(min_patches / (pw * ph)), *caps)
    nw, nh = max(1, int(width * scale)), max(1, int(height * scale))
    nw, nh = min(nw, limit_one_side * patch), min(nh, limit_one_side * patch)
    f = merge * patch
    th, tw = (nh + (f - nh % f) % f) // f, (nw + (f - nw % f) % f) // f
    KIMI_LAST_SIZE[0] = f"{tw * f}x{th * f}"
    return th * tw


KIMI_LAST_SIZE = [None]


def _kimi_num_tokens(text: str) -> int:
    """Kimi's tiktoken pattern splits digit runs into chunks of up to three (\\p{N}{1,3}); letters are one piece."""
    import re
    return sum(math.ceil(len(m) / 3) if m.isdigit() else 1 for m in re.findall(r"\d+|[^\d]+", text))


def kimi(w, h, src_fps, n, sample_fps=8.0, chunk=4, max_frames=None, patch_limit=16384, min_patches=0, max_tokens=None,
         size_text=False, video_budget=False):
    sampled = max(round(n * min(sample_fps, src_fps) / src_fps), 1)
    if max_frames:
        sampled = min(sampled, max_frames)
    units = math.ceil(sampled / chunk)
    # in_patch_limit_video (655,360 patches) is split per 4-frame chunk in vLLM; video_budget applies it
    # across every sampled frame of the video instead, so long videos get smaller frames.
    per_frame_limit = round(655360 / (sampled if video_budget else chunk))
    per = kimi_resize(w, h, in_patch_limit=min(per_frame_limit, patch_limit), min_patches=min_patches)
    if max_tokens:                      # hypothetical provider cap on a video's visual tokens: drop chunks
        units = max(1, min(units, max_tokens // per))
        sampled = min(sampled, units * chunk)
    # Some Kimi deployments write the frame size into the video prompt ("1920x1080"), once per video.
    digits = _kimi_num_tokens(f"{w}x{h}") if size_text else 0
    return {"frames": sampled, "units": units, "visual": units * per, "per_unit": per, "digits": digits,
            "size": KIMI_LAST_SIZE[0]}


# GLM-5.3-Flash: transformers glm5_next video processor (2 fps, 2-frame groups, token budget).
@lru_cache(maxsize=None)
def glm_smart_resize(num_frames, height, width, temporal=2, factor=28, min_tokens=16, max_tokens=240000):
    ppt = temporal * factor ** 2
    min_px, max_px = min_tokens * ppt, max_tokens * ppt
    align = lambda v: math.ceil(v / factor) * factor  # noqa: E731
    frames = max(temporal, round(num_frames / temporal) * temporal)
    ah, aw = align(height), align(width)
    if frames * ah * aw < min_px:
        sc = math.sqrt(min_px / (num_frames * height * width))
        ah, aw = align(max(1, math.ceil(height * sc))), align(max(1, math.ceil(width * sc)))
    if frames * ah * aw > max_px:
        lo, hi, best = 1, height, (factor, factor)
        while lo <= hi:
            ch = (lo + hi) // 2
            h2, w2 = align(ch), align(max(1, math.floor(width * ch / height)))
            if frames * h2 * w2 <= max_px:
                best, lo = (h2, w2), ch + 1
            else:
                hi = ch - 1
        ah, aw = best
    return ah, aw


@lru_cache(maxsize=None)
def glm_sample(total, src_fps, target_fps=2.0, max_frames=2048) -> tuple:
    duration = total / src_fps
    max_seconds = int(duration)
    extract_t = min(int(duration * target_fps), max_frames)
    if total < extract_t:
        idx = np.linspace(0, total - 1, extract_t, dtype=int).tolist()
    else:
        idx, cur = [], 0.0
        for i in range(total):
            if i / src_fps >= cur:
                cur += 1 / target_fps
                idx.append(i)
                if cur >= max_seconds:
                    break
    if len(idx) < extract_t:
        s, e = (idx[0], idx[-1]) if idx else (0, max(total - 1, 0))
        idx = np.linspace(s, e, extract_t, dtype=int).tolist()
    elif len(idx) > extract_t:
        idx = np.linspace(0, total - 1, extract_t, dtype=int).tolist()
    uniq = list(dict.fromkeys(idx))
    if len(uniq) & 1:
        uniq.append(uniq[-1])
    return tuple(uniq)


@lru_cache(maxsize=None)
def glm_text(total, src_fps, target_fps, fixed_frames):
    idx = list(_linspace(total - 1, fixed_frames, 'floor')) if fixed_frames else list(glm_sample(total, src_fps, target_fps))
    if len(idx) % 2:
        idx.append(idx[-1])
    return len(idx), text_tokens('glm', tuple(f"{i / src_fps:.1f} seconds" for i in idx[::2]))


def glm(w, h, src_fps, n, fps=2.0, budget=240000, fixed_frames=None):
    fixed = fixed_frames if (fixed_frames and n >= fixed_frames) else None
    frames, text = glm_text(n, src_fps, fps, fixed)
    if frames < 2:
        return None
    rh, rw = glm_smart_resize(frames, h, w, max_tokens=budget)
    units = (frames + frames % 2) // 2
    per = (rh // 28) * (rw // 28)
    return {"frames": frames, "units": units, "visual": units * per, "per_unit": per, "digits": text, "size": f"{rw}x{rh}"}


# Qwen3.6 / Qwen3.8: transformers qwen3_vl video processor (2 fps, 4..768 frames, 32 px tokens,
# 2-frame groups, total pixel budget `longest_edge`; optional per-frame cap as in qwen-vl-utils).
@lru_cache(maxsize=None)
def qwen_smart_resize(num_frames, height, width, factor=32, min_pixels=4096, max_pixels=25165824):
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    t_bar = round(num_frames / 2) * 2
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@lru_cache(maxsize=None)
def qwen_digits(n, src_fps, frames):
    idx = list(_linspace(n - 1, frames))
    if len(idx) % 2:
        idx.append(idx[-1])
    return text_tokens('qwen', tuple(f"<{(idx[i] + idx[i + 1]) / 2 / src_fps:.1f} seconds>" for i in range(0, len(idx), 2)))


def qwen(w, h, src_fps, n, fps=2.0, min_frames=4, max_frames=768, budget=25165824, capped=False,
         max_video_tokens=768, min_pixels=4096, fixed_frames=None, align=1):
    if fixed_frames and n >= fixed_frames:      # provider samples a fixed number of frames per video
        frames = fixed_frames
    else:
        frames = min(max(int(n / src_fps * fps), min_frames), max_frames, n)
        if align > 1:                           # provider rounds the frame count down to a multiple
            frames = max(min_frames, frames // align * align)
    if frames < 2:
        return None
    max_px = budget
    if capped:
        ppf = max(min(max_video_tokens * 32 * 32, budget // frames), int(min_pixels * 1.05))
        max_px = ppf * frames
    rh, rw = qwen_smart_resize(frames, h, w, min_pixels=min_pixels, max_pixels=max_px)
    units = (frames + frames % 2) // 2
    per = (rh // 32) * (rw // 32)
    return {"frames": frames, "units": units, "visual": units * per, "per_unit": per,
            "digits": qwen_digits(n, src_fps, frames), "size": f"{rw}x{rh}"}


# Gemma 4: transformers gemma4 video processor (32 frames, 70 soft tokens, pooling 3, patch 16).
def gemma_size(height, width, patch=16, max_soft=70, pool=3):
    target_px = max_soft * pool ** 2 * patch ** 2
    f = math.sqrt(target_px / (height * width))
    m = pool * patch
    th, tw = int(math.floor(f * height / m)) * m, int(math.floor(f * width / m)) * m
    GEMMA_LAST_SIZE[0] = f"{tw}x{th}"
    return (th // patch) * (tw // patch) // pool ** 2


GEMMA_LAST_SIZE = [None]


def gemma(w, h, src_fps, n, num_frames=32, max_soft=70, fps=None, clamp=False):
    frames = max(1, int(n / src_fps * fps)) if fps else num_frames
    if frames > n:
        if not clamp:
            return None  # the reference raises: more frames requested than the video has
        frames = n       # serving stacks that clamp to the frames available
    per = gemma_size(h, w, max_soft=max_soft)
    return {"frames": frames, "units": frames, "visual": frames * per, "per_unit": per, "digits": 0, "size": GEMMA_LAST_SIZE[0]}


# Nemotron 3.5 Super VL: processing_nemotron_h_omni (video tile near 1024 patches, 2-frame tubelets,
# "Frame i sampled at t seconds" labels), vLLM nano_nemotron_vl + EVS (video_pruning_rate 0.7 in
# config.json), vLLM VideoBackend sampling (num_frames / fps, linspace).
def nemotron_tile(width, height, target=1024, divisor=2):
    a = width / max(height, 1)
    ph, pw = max(round(math.sqrt(target / a)), 1), max(round(math.sqrt(target * a)), 1)
    rh, rw = ph % divisor, pw % divisor
    phu, phd = ph + (divisor - rh if rh else 0), ph - rh
    pwu, pwd = pw + (divisor - rw if rw else 0), pw - rw
    if phu * pwu <= target:
        ph, pw = phu, pwu
    else:
        ph, pw = max(divisor, phd), max(divisor, pwd)
    return (ph * pw) // divisor ** 2, f"{pw * 16}x{ph * 16}"


@lru_cache(maxsize=None)
def nemotron_digits(n, src_fps, k):
    idx = list(range(k)) if k == n else np.linspace(0, n - 1, k, dtype=int).tolist()
    ms = int(1000.0 / src_fps)
    parts = [f"{'Frame' if i % 2 == 0 else 'frame'} {i + 1} sampled at {idx[i] * ms / 1000.0:.2f} seconds" for i in range(k)]
    labels = tuple(' and '.join(parts[g:g + 2]) + ': ' for g in range(0, k, 2))
    return text_tokens('nemotron', labels)


def nemotron(w, h, src_fps, n, num_frames=-1, fps=-1.0, target=1024, q=0.7, T=2):
    k = n if num_frames <= 0 else min(num_frames, n)
    if fps > 0:
        k = min(k, math.floor(n / src_fps * fps))
    k = max(1, k)
    per, size = nemotron_tile(w, h, target)
    tub = math.ceil(k / T)
    visual = max(per, int(per * tub * (1 - q))) if q > 0 else per * tub
    return {"frames": k, "units": tub, "visual": visual, "per_unit": per, "digits": nemotron_digits(n, src_fps, k), "size": size}


# Muse Spark (open): vLLM rust muse_spark processor (2 fps connector, 600-frame cap, 2-frame clips,
# least-distortion grid of 28 px token patches, <=256 tokens per frame, "Time: N.NNs" labels).
@lru_cache(maxsize=None)
def muse_grid(img_h, img_w, tp=28, min_tok=1, max_tok=256):
    i_nph, i_npw = img_h / tp, img_w / tp
    ratio = i_npw / i_nph
    if i_nph * i_npw > max_tok:
        nph = math.sqrt(max_tok / ratio)
        i_nph, i_npw = nph, nph * ratio
    elif i_nph * i_npw < min_tok:
        nph = math.sqrt(min_tok / ratio)
        i_nph, i_npw = nph, nph * ratio
    cands = sorted({(a, b) for a in (math.floor(i_nph), math.ceil(i_nph)) for b in (math.floor(i_npw), math.ceil(i_npw))})
    ok = [(a, b) for a, b in cands if a > 0 and b > 0 and min_tok <= a * b <= max_tok]
    if ok:
        a, b = min(ok, key=lambda g: abs(g[0] / g[1] - img_h / img_w))
        return a * b, f"{b * tp}x{a * tp}"
    best = None
    rn = round(i_npw)
    for a in range(1, max_tok + 1):
        up, lo = max_tok // a, -(-min_tok // a)
        if up < lo:
            continue
        b = min(max(rn, lo), up)
        l2 = ((a - i_nph) * tp) ** 2 + ((b - i_npw) * tp) ** 2
        if best is None or l2 < best[2]:
            best = (a, b, l2)
    return best[0] * best[1], f"{best[1] * tp}x{best[0] * tp}"


@lru_cache(maxsize=None)
def muse_frames(n, src_fps, fps, max_frames):
    k = max(1, min(n, math.floor(n / src_fps * fps)))     # connector (vLLM VideoBackend) at `fps`
    target = max(2, min(k, max_frames) // 2 * 2)
    idx = list(range(k)) if k <= target else list(_linspace(k - 1, target))
    aligned = max(2, len(idx) // 2 * 2)
    idx = (idx + [idx[-1]] * 2)[:aligned]
    clips = aligned // 2
    return aligned, clips, text_tokens('muse', tuple(f"Time: {idx[2 * c] / fps:.2f}s" for c in range(clips)))


def muse(w, h, src_fps, n, fps=2.0, max_frames=600, max_tok_frame=256, budget=None):
    aligned, clips, digits = muse_frames(n, src_fps, fps, max_frames)
    per, size = muse_grid(h, w, max_tok=(budget // clips if budget else max_tok_frame))
    return {"frames": aligned, "units": clips, "visual": clips * per, "per_unit": per, "digits": digits, "size": size}


# Gemini (closed; documented): a fixed frame rate at a fixed token count per frame.
def gemini(w, h, src_fps, n, fps=1.0, per=66, rounding="ceil"):
    d = n / src_fps * fps
    frames = max(1, {"ceil": math.ceil, "floor": math.floor, "round": round}[rounding](d - 1e-9))
    return {"frames": frames, "units": frames, "visual": frames * per, "per_unit": per, "digits": 0}


FAMILIES: dict[str, dict[str, Any]] = {
    "kimi": {"fn": kimi, "ref": {}, "d": 0, "unit": "4-frame chunk",
             "grid": lambda: [dict(sample_fps=f, max_frames=m, patch_limit=p, min_patches=mp)
                              for f in (1, 2, 4, 8) for m in (None, 16, 32, 48, 60, 64, 96, 120, 128)
                              for p in (4096, 8192, 16384) for mp in (0, 256, 576, 1024, 4096)] +
                             [dict(sample_fps=f, max_tokens=t, size_text=st) for f in (2, 4, 8) for t in range(8192, 200001, 128)
                              for st in (False, True)] +
                             [dict(sample_fps=f, video_budget=True, size_text=st) for f in (2, 4, 8) for st in (False, True)]},
    "glm": {"fn": glm, "ref": {}, "d": 1, "unit": "2-frame group",
            "grid": lambda: [dict(fps=f, budget=b, fixed_frames=x) for f in (1, 2, 4) for x in (None, 16, 32, 64)
                             for b in list(range(2000, 60000, 250)) + list(range(60000, 400001, 1000))]},
    "qwen": {"fn": qwen, "ref": {}, "d": 1, "unit": "2-frame group",
             "grid": lambda: [dict(fps=f, min_frames=mn, max_frames=m, capped=v is not None, max_video_tokens=v or 768, budget=b)
                              for f in (1, 2, 4) for mn in (2, 4, 8) for m in (8, 16, 32, 64, 128, 256, 768)
                              for v in (None, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 1024, 1280)
                              for b in (25165824, 10 ** 10)] +
                             [dict(fps=f, max_frames=m, capped=False, budget=b * 2048)
                              for f in (1, 2, 4) for m in (64, 128, 256, 768)
                              for b in list(range(32, 1024, 8)) + list(range(1024, 32768, 64)) + list(range(32768, 524289, 1024))] +
                             [dict(fixed_frames=x, budget=b * 2048) for x in (16, 32, 64) for b in range(10240, 14337, 64)]},
    "gemma": {"fn": gemma, "ref": {}, "d": 0, "unit": "frame",
              "grid": lambda: [dict(num_frames=nf, max_soft=s, clamp=c) for nf in (8, 16, 24, 32, 48, 64, 96, 128)
                               for s in (70, 140, 280, 560, 1120) for c in (False, True)] +
                              [dict(fps=f, max_soft=s) for f in (0.5, 1, 2) for s in (70, 140, 280, 560, 1120)]},
    "nemotron": {"fn": nemotron, "ref": {"num_frames": 32}, "d": 1, "unit": "2-frame tubelet",
                 "grid": lambda: [dict(num_frames=nf, fps=f, target=t, q=q)
                                  for nf in (8, 16, 24, 32, 48, 64, 96, 128, 256, 512, -1)
                                  for f in (-1, 0.5, 1, 2, 4) for t in (256, 512, 1024, 2048, 4096) for q in (0.0, 0.5, 0.7, 0.75)]},
    "muse": {"fn": muse, "ref": {}, "d": 1, "unit": "2-frame clip",
             "grid": lambda: [dict(fps=f, max_frames=m, max_tok_frame=t) for f in (0.5, 1, 2, 4)
                              for m in (16, 32, 48, 64, 96, 128, 192, 256, 300, 600, 1200)
                              for t in (16, 32, 64, 96, 128, 144, 192, 256, 384, 512, 1024)] +
                             [dict(fps=f, max_frames=m, budget=b) for f in (1, 2) for m in (64, 128, 256, 600)
                              for b in (2048, 4096, 8192, 16384)]},
    "gemini": {"fn": gemini, "ref": {}, "d": 0, "unit": "frame",
               "grid": lambda: [dict(fps=f, per=p, rounding=r) for f in (0.5, 1, 2) for p in range(32, 300)
                                for r in ("ceil", "floor", "round")]},
}


# Most per-unit marker text the model's video prompt can add, beyond the timestamp labels counted
# separately (begin/end markers, separators): a fit needing more is not this format.
K_MAX = {"kimi": 20, "glm": 10, "qwen": 10, "muse": 10, "nemotron": 40, "gemma": 15, "gemini": 5}


def family_of(name: str) -> Optional[str]:
    n = name.lower()
    for key in ("kimi", "glm", "qwen", "gemma", "nemotron", "muse", "gemini"):
        if n.startswith(key):
            return key
    return None


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def predict(fam: str, knobs: dict, clip: dict) -> Optional[dict]:
    return FAMILIES[fam]["fn"](clip["width"], clip["height"], clip["fps"], clip["frames"], **knobs)


def check(fam: str, knobs: dict, clips: list[dict], d: int, tol: float = 1.0) -> Optional[dict]:
    """Fit k, a for this hypothesis; accept only if every clip is within tol tokens."""
    rows, ys, preds = [], [], []
    for c in clips:
        p = predict(fam, knobs, c)
        if p is None:            # the hypothesis says this clip errors, but the provider billed it
            return None
        preds.append(p)
        rows.append(p["units"])
        ys.append(c["billed"] - p["visual"] - d * p["digits"])
    x, y = np.array(rows, float), np.array(ys, float)
    if np.ptp(x) == 0:           # units constant: k and a can't be separated
        k, a = 0.0, float(y.mean())
    else:
        k, a = np.polyfit(x, y, 1)
    if k < -0.5 or k > K_MAX.get(fam, 80):     # k is per-unit marker text: bounded by the model's prompt format
        return None
    res = y - (k * x + a)
    worst = float(np.max(np.abs(res)))
    if worst > tol:
        return None
    return {"knobs": knobs, "d": d, "k": round(float(k), 2), "a": round(float(a), 2), "max_residual": round(worst, 2),
            "frames": [p["frames"] for p in preds], "units": [p["units"] for p in preds],
            "per_unit": [p["per_unit"] for p in preds], "size": [p.get("size") for p in preds]}


def analyse(route: dict) -> dict:
    fam = route["family"]
    clips = [c for c in route["clips"] if c.get("billed") is not None]
    out: dict[str, Any] = {"n_clips": len(clips), "n_failed": len(route["clips"]) - len(clips)}
    reps = route.get("repeats", [])
    out["repeat_identical"] = all(r.get("billed") == r.get("first") for r in reps) if reps else None
    if len(clips) < 4:
        out["verdict"] = "too few clips"
        return out
    F = FAMILIES[fam]
    ref = check(fam, F["ref"], clips, F["d"])
    acc = [ref] if ref else []
    for knobs in F["grid"]():
        for d in dict.fromkeys((F["d"], 1 - F["d"])):
            r = check(fam, knobs, clips, d)
            if r:
                acc.append(r)
                break
    if not acc:
        out["verdict"] = "unexplained"
        best = None
        for knobs in [F["ref"]] + F["grid"]():
            for d in (0, 1):
                r = check(fam, knobs, clips, d, tol=float("inf"))
                if r and (best is None or r["max_residual"] < best["max_residual"]):
                    best = r
        out["closest"] = best
        return out
    frames_sets = {tuple(r["frames"]) for r in acc}
    units_sets = {tuple(r["units"]) for r in acc}
    if ref:
        # The reference fits; it is only "determined" if no other setting fits with different frame counts.
        same = len(frames_sets) == 1 and len(units_sets) == 1
        out.update(verdict="matches reference" + ("" if same else " (other settings also fit)"), accepted=[ref] + acc[1:3],
                   accepted_count=len(acc), determined=same, frame_sets=len(frames_sets),
                   accepted_all=[{k: r[k] for k in ("knobs", "d", "k", "a")} for r in acc])
        return out
    out["accepted_count"] = len(acc)
    out["accepted"] = acc[:3]
    out["accepted_all"] = [{k: r[k] for k in ("knobs", "d", "k", "a")} for r in acc]
    keys = sorted({k for r in acc for k in r["knobs"]})
    rng = {}
    for k in keys:
        vals = sorted({r["knobs"].get(k) for r in acc}, key=lambda v: (v is None, v if not isinstance(v, bool) else int(v)))
        rng[k] = vals if len(vals) <= 8 else [vals[0], "…", vals[-1], f"({len(vals)} values)"]
    out["knob_ranges"] = rng
    out["determined"] = len(frames_sets) == 1 and len(units_sets) == 1
    out["frame_sets"] = len(frames_sets)
    out["verdict"] = "provider settings found" + ("" if out["determined"] else " (frame counts ambiguous)")
    return out


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def run(args) -> Path:
    import aiohttp
    from dotenv import dotenv_values, load_dotenv
    import probe as P

    load_dotenv(P.HERE / ".env", override=False)
    secrets = {k: v for k, v in dotenv_values(P.HERE / ".env.local").items() if v}
    files = [Path(p) for p in args.targets]
    out_dir = P.RESULTS / f"curve-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    P.LOG = P.RequestLog(out_dir / "requests.jsonl")
    fx = P.Fixtures()

    def ensure(w, h, fps, secs) -> None:
        key = f"probe/curve/{clip_name(w, h, fps, secs)}"
        n = clip_frames(fps, secs)
        try:
            fx.s3.client.head_object(Bucket=fx.s3.bucket, Key=key)
            return
        except Exception:  # noqa: BLE001 - build it
            pass
        font = next((f for f in P.FONT_CANDIDATES if os.path.exists(f)), None)
        fontarg = f"fontfile={font}:" if font else ""
        with tempfile.TemporaryDirectory() as d:
            o = os.path.join(d, "c.mp4")
            vf = (f"drawtext={fontarg}text='%{{eif\\:n\\:d\\:5}}':fontsize={max(24, int(h * 0.4))}:fontcolor=white:"
                  "x=(w-text_w)/2:y=(h-text_h)/2")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}",
                            "-frames:v", str(n), "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                            "-crf", "28", "-movflags", "+faststart", o], check=True)
            s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                           "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json", o],
                                          capture_output=True, text=True).stdout)["streams"][0]
            assert (int(s["nb_read_frames"]), s["width"], s["height"], s["r_frame_rate"]) == (n, w, h, f"{fps}/1"), (key, s)
            fx.s3.upload(o, key, "video/mp4")

    spec = [tuple(int(x) if i < 3 else float(x) for i, x in enumerate(c.split("x"))) for c in args.ladder.split(",")] \
        if args.ladder else LADDER
    ladder = [dict(width=w, height=h, fps=f, seconds=s, frames=clip_frames(f, s), clip=clip_name(w, h, f, s))
              for w, h, f, s in spec if not args.max_seconds or s <= args.max_seconds]
    # Register the reference prediction for every family and clip before any request is sent.
    prereg = {fam: [{"clip": c["clip"], "reference": predict(fam, FAMILIES[fam]["ref"], c)} for c in ladder] for fam in FAMILIES}
    (out_dir / "predictions.json").write_text(json.dumps({"ladder": ladder, "reference": prereg}, indent=1, default=str))
    for c in ladder:
        ensure(c["width"], c["height"], c["fps"], c["seconds"])
    P.console.print(f"clips verified with ffprobe · reference predictions registered → {out_dir / 'predictions.json'}")

    async def route(http, t, sem) -> dict:
        res: dict[str, Any] = {"target": t.name, "model": t.model, "pin": t.pin, "family": family_of(t.name),
                               "clips": [], "repeats": []}
        async with sem:
            P.CTX.set({"dimension": "frame_curve", "value": "text_baseline"})
            base = await P.call(http, t, [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
                                stream=False, max_tokens=8, timeout=180)
            res["baseline_prompt_tokens"] = base.prompt_tokens
            if not base.ok or not base.prompt_tokens:
                res["error"] = f"baseline: {base.kind} {base.message[:200]}"
                return res

            async def send(c: dict) -> dict:
                key = f"probe/curve/{c['clip']}"
                for attempt in range(2):
                    url = await asyncio.to_thread(fx.data_uri if t.video_inline else fx.s3.presign, key)
                    P.CTX.set({"dimension": "frame_curve", "value": c["clip"] + (" inline" if t.video_inline else "")})
                    r = await P.call(http, t, [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT}, {"type": "video_url", "video_url": {"url": url}}]}],
                        stream=False, max_tokens=8, timeout=args.timeout)
                    if not r.ok and r.kind in P.INLINE_RETRY and not t.video_inline and attempt == 0:
                        t.video_inline = True      # URL refused or filtered out: send the same clip inline
                        res["inline"] = True
                        continue
                    break
                row = {**c, "ok": r.ok, "latency_s": round(r.seconds, 1), "request_id": r.request_id}
                if r.ok and r.prompt_tokens:
                    row["billed"] = r.prompt_tokens - base.prompt_tokens
                else:
                    row.update(kind=r.kind, message=r.message[:300])
                return row

            for i, c in enumerate(ladder):
                row = await send(c)
                res["clips"].append(row)
                P.console.print(f"[dim]{datetime.now():%H:%M:%S}[/] {t.name:50} {c['clip']:34} "
                                f"{row.get('billed', '✗ ' + str(row.get('kind'))):>9}", highlight=False)
                if i == 0 and not row["ok"] and row.get("kind") in ("no_video", "no_route", "url_unsupported", "auth_or_billing", "credit_guard"):
                    res["error"] = f"first clip: {row.get('kind')} {str(row.get('message', ''))[:160]}"
                    return res
            for w, h, f, s in REPEAT:
                c = next((x for x in ladder if (x["width"], x["height"], x["fps"], x["seconds"]) == (w, h, f, s)), None)
                if c is None:          # clip excluded by --max-seconds
                    continue
                first = next((x.get("billed") for x in res["clips"] if x["clip"] == c["clip"]), None)
                if first is not None:
                    row = await send(c)
                    res["repeats"].append({"clip": c["clip"], "first": first, "billed": row.get("billed")})
        return res

    async def main_async() -> list[dict]:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as http:
            targets = [t for t in await P.load_targets(http, files, secrets, args.only, args.exclude) if family_of(t.name)]
            P.GUARD = P.CreditGuard(http, secrets.get("OPENROUTER_API_KEY"), args.openrouter_floor)
            left = await P.GUARD.remaining()
            P.console.print(f"frame curve · {len(targets)} routes · OpenRouter balance {left}\n[dim]→ {out_dir}[/]")
            lanes: dict[str, asyncio.Semaphore] = {}
            sems = [lanes.setdefault(f"{t.service}:{t.pin}" if t.service == "openrouter" else t.service,
                                     asyncio.Semaphore(2)) for t in targets]
            res = await asyncio.gather(*(route(http, t, s) for t, s in zip(targets, sems)))
            P.console.print(f"[dim]OpenRouter balance now {await P.GUARD.remaining()}[/]")
            return list(res)

    results = asyncio.run(main_async())
    path = out_dir / "curve.json"
    path.write_text(json.dumps(results, indent=1, default=str))
    return path


def report(path: Path) -> None:
    results = json.loads(path.read_text())
    summary = []
    for r in results:
        line = {"target": r["target"], "family": r["family"]}
        line.update({"verdict": r["error"]} if r.get("error") else analyse(r))
        summary.append(line)
        print(f"\n{r['target']}: {line['verdict']}  · repeats identical: {line.get('repeat_identical')}"
              f" · {line.get('n_clips')} clips (+{line.get('n_failed')} failed)")
        acc = (line.get("accepted") or [None])[0]
        if not acc and line.get("closest"):
            b = line["closest"]
            print(f"   closest: {b['knobs']} k={b['k']} a={b['a']} max|residual| {b['max_residual']}")
        if acc:
            print(f"   {acc['knobs'] or 'reference settings'}  k={acc['k']} a={acc['a']} d={acc['d']}  max|residual| {acc['max_residual']}"
                  f"  · hypotheses accepted: {line.get('accepted_count')} · frames determined: {line.get('determined')}")
            clips = [c for c in r["clips"] if c.get("billed") is not None]
            print("   " + "  ".join(f"{c['seconds']:g}s/{c['height']}p{c['fps']}→{f}f,{u}u×{p}" for c, f, u, p in
                                     zip(clips, acc["frames"], acc["units"], acc["per_unit"])))
            if not line.get("determined") and line.get("knob_ranges"):
                print(f"   knob ranges: {line['knob_ranges']}")
        for c in r.get("clips", []):
            if c.get("billed") is None:
                print(f"   failed {c['clip']}: {c.get('kind')} {str(c.get('message', ''))[:120]}")
    (path.parent / "analysis.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n→ {path.parent / 'analysis.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", action="append", default=[])
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--openrouter-floor", type=float, default=2.0)
    ap.add_argument("--ladder", help="custom clips WxHxFPSxSECONDS, comma-separated (default: the standard ladder)")
    ap.add_argument("--max-seconds", type=float, help="skip ladder clips longer than this (providers that time out on long video)")
    ap.add_argument("--analyze", help="re-analyse an existing curve.json without sending requests")
    args = ap.parse_args()
    if args.analyze:
        report(Path(args.analyze))
        return
    if not args.targets:
        sys.exit("--targets is required")
    report(run(args))


if __name__ == "__main__":
    main()
