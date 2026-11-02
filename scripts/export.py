"""Turn a probe run into vlm.fyi's public data files.

    python scripts/export.py --probe path/to/probe.json --catalog path/to/models.json

Reads the raw probe output (one record per provider + model, see
github.com/... /probe.py) and writes:

  data/capabilities.json         the current dataset (schema below)
  data/snapshots/<date>.json     a dated copy, kept forever
  data/changes.json              what changed versus the previous snapshot

Raw evidence is scrubbed before it is written: URLs (including signed storage
links), request ids, user / account ids and API-key-looking strings are removed.
Private deployments (see PRIVATE_HOSTS) are left out.

Standard library only, so it runs anywhere.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCHEMA = "https://vlm.fyi/schema/capabilities-v1.json"
PROBE_VERSION = "2"

# Endpoints that are someone's own deployment, not a public API.
PRIVATE_HOSTS = ("trycloudflare.com", "localhost", "127.0.0.1")
# Probe targets left out of the public data, with the reason.
EXCLUDED_TARGETS = {
    # Older Modal endpoint, superseded by the -1017 one. It also "accepted" video
    # for a model with no video input, which suggests the input was ignored.
    "deepseek-v4.1-flash · modal": "superseded endpoint",
}
# Hostnames can carry account names (e.g. <workspace>--ep-<name>.<region>.modal.direct);
# the dataset never stores endpoint URLs, and the leak check below enforces it.
SERVICE_HOSTS = {"openrouter": "openrouter", "fireworks": "fireworks", "together": "together",
                 "jarvislabs": "jarvislabs", "modal.direct": "modal"}

PROVIDERS = {
    "fireworks": ("Fireworks", "https://fireworks.ai"),
    "together": ("Together AI", "https://together.ai"),
    "jarvislabs": ("Jarvislabs", "https://jarvislabs.ai"),
    "modal": ("Modal", "https://modal.com"),
    "openrouter": ("OpenRouter", "https://openrouter.ai"),
    "alibaba": ("Alibaba Cloud", "https://www.alibabacloud.com"),
    "z-ai": ("Z.ai", "https://z.ai"),
    "google-ai-studio": ("Google AI Studio", "https://aistudio.google.com"),
    "minimax": ("MiniMax", "https://www.minimax.io"),
}

# OpenRouter upstream tags -> display names. Variants after "/" (fp8, global,
# flex…) are shown in parentheses: "chutes/fp8" -> "Chutes (fp8)".
UPSTREAM_NAMES = {
    "alibaba": "Alibaba Cloud", "z-ai": "Z.ai", "google-ai-studio": "Google AI Studio", "google-vertex": "Google Vertex",
    "minimax": "MiniMax", "fireworks": "Fireworks", "together": "Together AI", "novita": "Novita", "chutes": "Chutes",
    "reka": "Reka", "parasail": "Parasail", "coreweave": "CoreWeave", "deepinfra": "DeepInfra", "akashml": "AkashML",
    "venice": "Venice", "dekallm": "DekaLLM", "darkbloom": "DarkBloom", "near-ai": "NEAR AI", "ionstream": "IonStream",
    "modal": "Modal", "wafer": "Wafer", "tenstorrent": "Tenstorrent", "mancer": "Mancer", "modelrun": "ModelRun",
    "phala": "Phala", "cloudflare": "Cloudflare", "cerebras": "Cerebras", "atlas-cloud": "Atlas Cloud",
    "siliconflow": "SiliconFlow",
}


def upstream_name(tag: str) -> str:
    base, *variants = tag.split("/")
    name = UPSTREAM_NAMES.get(base, base.replace("-", " ").title())
    return f"{name} ({', '.join(variants)})" if variants else name


SIZE_KINDS = {"request_bytes", "decoded_bytes", "media_bytes", "download_bytes", "body_timeout"}
SOFT_KINDS = {"timeout", "transient", "auth_or_billing", "credit_guard"}
IGNORED_BELOW_TOK_PER_MIN = 50   # a video that adds fewer tokens than this was not seen


# Limits a run produced outside the probe (agent sessions), for pairs where the
# probe couldn't measure them. Keyed by the probe target name.
OBSERVATIONS_FILE = DATA / "observations.json"

LIMITS = {
    # key: (probe dimension, unit, label)
    "images_per_request": ("image_count", "count", "Images per request"),
    "frames_768x432": ("frames_768", "count", "768×432 frames per request"),
    "inline_request_bytes": ("inline_bytes", "bytes", "Inline request body"),
    "url_image_bytes": ("url_bytes", "bytes", "Images by URL (total)"),
    "video_file_bytes": ("video_bytes", "bytes", "Video file size"),
}
MB = 1_000_000


# --------------------------------------------------------------------------
# Scrubbing
# --------------------------------------------------------------------------

_SCRUB = [
    (re.compile(r"https?://\S+?(?=[\s\"'\\)\]}>,]|$)"), "<url>"),
    (re.compile(r'(\\?"(?:request_id|id|user_id|account_id|organization|org_id)\\?"\s*:\s*\\?")[^"\\]+'), r"\1<redacted>"),
    (re.compile(r"\b(?:sk|fw|tgp|wk|pk|rk)[-_][A-Za-z0-9_.\-]{12,}"), "<key>"),
    (re.compile(r"\buser_[A-Za-z0-9]{10,}\b"), "<user>"),
    (re.compile(r"\b[0-9a-f]{24,}\b"), "<id>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
]


def scrub(text: Optional[str], limit: int = 400) -> Optional[str]:
    if not text:
        return None
    for pat, rep in _SCRUB:
        text = pat.sub(rep, text)
    # Prefer the human-readable message inside a JSON error, if there is one.
    msgs = re.findall(r'\\?"message\\?"\s*:\s*\\?"((?:[^"\\]|\\.)+?)\\?"', text)
    msgs = [m for m in msgs if m.lower() not in ("provider returned error", "error")]
    if msgs:
        text = msgs[-1]
    text = text.replace('\\"', '"').replace("\\n", " ").strip()
    return text[:limit]


# --------------------------------------------------------------------------
# Model identity
# --------------------------------------------------------------------------

_QUANT = re.compile(r"^(nvfp\d+|fp\d+|bf16|int\d+|w\d+a\d+|q\d+(?:_[a-z0-9]+)*|\d+bit|awq|gptq|gguf|mlx)$", re.I)


def core_key(model_id: str) -> str:
    m = model_id.strip().lower().split("/")[-1].split(":")[0]
    m = re.sub(r"(?<=\d)p(?=\d)", ".", m)            # Fireworks writes 4.1 as 4p1
    parts = m.split("-")
    while len(parts) > 1 and _QUANT.match(parts[-1]):
        parts.pop()
    return "-".join(parts).replace("-", "")


def catalog_match(model_id: str, catalog: list[dict], overrides: list[dict]) -> Optional[dict]:
    key = core_key(model_id)
    for o in overrides:
        if key in {core_key(a) for a in [o["id"], *o.get("aliases", [])]}:
            return {"id": o["id"], "name": f"{o['vendor']}: {o['name']}",
                    "architecture": {"input_modalities": o.get("input_modalities")}}
    for e in catalog:
        if ":" not in e["id"] and core_key(e["id"]) == key:
            return e
    return None


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-")


# --------------------------------------------------------------------------
# Probe results -> public limits
# --------------------------------------------------------------------------

def to_limit(res: Optional[dict], unit: str, *, scale: float = 1.0) -> dict:
    """One probe search result -> {status, value, ...}.

    status: measured      largest passing value found, with the first failure
            at_least      no failure up to the probe's cap (value = cap)
            inconclusive  the search stopped on a timeout or transient error
            stated        not searched to the end, but the provider's error states it
            unsupported   the input type isn't accepted at all
            not_tested
    """
    if not res:
        return {"status": "not_tested"}
    conv = lambda v: int(round(v * scale))  # noqa: E731
    lo, hi = res.get("max_ok"), res.get("fail_at")
    if res.get("inconclusive"):
        out = {"status": "inconclusive", "reason": res["inconclusive"]}
        if lo:
            out["at_least"] = conv(lo)
        return out
    if hi is None:
        return {"status": "at_least", "value": conv(lo)} if lo else {"status": "not_tested"}
    out = {"status": "measured", "value": conv(lo) if lo else 0, "fails_at": conv(hi),
           "failure": res.get("fail_kind")}
    if res.get("stated_limit"):
        out["stated"] = int(res["stated_limit"])
        if unit == "bytes" and res.get("fail_kind") == "download_bytes" and out["stated"] < 10_000:
            out["stated"] *= MB   # stated in MB ("cannot exceed 30MB")
    else:
        m = re.search(r"exceeds (?:the )?limit (?:of )?(\d{3,})", res.get("message") or "")
        if m:
            out["stated"] = int(m.group(1))
    ev = scrub(res.get("message"))
    if ev:
        out["evidence"] = ev
    return out


def entry_from_target(t: dict, catalog: list[dict], overrides: list[dict], measured_at: str) -> Optional[dict]:
    base = t["base_url"].lower()
    if any(h in base for h in PRIVATE_HOSTS) or t["name"] in EXCLUDED_TARGETS:
        return None
    r = t["results"]
    if not (r.get("baseline") or {}).get("ok"):
        return None

    service = next((svc for host, svc in SERVICE_HOSTS.items() if host in base), None)
    if service is None:
        return None
    upstream = t.get("pin") if service == "openrouter" else None
    if service == "openrouter":
        provider_id = upstream or "auto"
        provider = {"id": slug(provider_id), "name": upstream_name(upstream) if upstream else "Auto-routed",
                    "via": "openrouter", "tag": upstream}
    else:
        provider = {"id": service, "name": PROVIDERS[service][0], "via": None}

    cat = catalog_match(t["model"], catalog, overrides)
    model = {
        "id": cat["id"] if cat else t["model"],
        "name": (cat or {}).get("name", t["model"]).split(": ", 1)[-1],
        "vendor": (cat or {}).get("name", "").split(": ", 1)[0] if cat and ": " in cat["name"] else None,
        "api_id": t["model"],
        "input_modalities": ((cat or {}).get("architecture") or {}).get("input_modalities"),
    }

    limits: dict[str, Any] = {}
    for key, (dim, unit, _label) in LIMITS.items():
        res = r.get(dim)
        if dim == "video_bytes":
            if res and res.get("video_input") is False:
                limits[key] = {"status": "unsupported", "evidence": scrub(res.get("message"))}
                continue
            res = (res or {}).get("bytes")
        limits[key] = to_limit(res, unit, scale=MB if unit == "bytes" else 1)
    f7 = r.get("frames_768") or {}
    img = limits["images_per_request"]
    if img["status"] in ("inconclusive", "not_tested") and f7.get("fail_kind") == "image_count" and f7.get("stated_limit"):
        limits["images_per_request"] = {"status": "stated", "value": int(f7["stated_limit"]),
                                        "evidence": scrub(f7.get("message")), "source": "768×432 frames probe"}
    if f7.get("tokens_per_frame"):
        limits["tokens_per_frame_768x432"] = {"status": "measured", "value": f7["tokens_per_frame"]}

    video = video_state(r)
    limits.update(density_limits(r.get("video_density"), t["model"], r.get("video_frames")))
    if video == "ignored":
        limits["video_tokens_per_min"]["status"] = "ignored"

    ii = r.get("image_input") or {}
    params = r.get("params") or {}
    to = r.get("tool_order") or {}
    entry = {
        "id": f"{provider['via'] + '-' if provider['via'] else ''}{provider['id']}/{slug(model['name'])}",
        "provider": provider,
        "model": model,
        "measured_at": measured_at,
        "limits": limits,
        "accepts": {
            "image_inline": ii.get("inline"),
            "image_url": ii.get("url"),
            "video": video,
            "request_fields": {k: v.get("ok") for k, v in params.items()},
        },
        "behaviour": {
            "strict_tool_message_order": (bool(to.get("contiguous_ok")) and not to.get("interleaved_ok")) if to else None,
        },
        "notes": [],
        "featured": not t.get("sweep_only", False),
    }
    if video == "ignored":
        entry["notes"].append("Accepts a video_url with HTTP 200, but the video adds no tokens: the model never sees it.")
    return entry


def video_state(r: dict) -> Optional[str]:
    """yes / no / ignored, from the density probe first (it shows whether the video
    reached the model), then from the file-size probe (accepted or rejected)."""
    d = r.get("video_density") or {}
    if d.get("visual_tokens_per_min") is not None:
        return "ignored" if d["visual_tokens_per_min"] < IGNORED_BELOW_TOK_PER_MIN else "yes"
    fail = next((x for x in d.get("runs", []) if not x.get("ok") and x.get("kind")), None)
    if fail and fail["kind"] not in SOFT_KINDS | SIZE_KINDS:
        return "no"
    vb = r.get("video_bytes") or {}
    if vb.get("video_input") is True:
        return "yes"
    if vb.get("video_input") is False and vb.get("kind") not in SOFT_KINDS:
        return "no"
    return None


def calibration_complete(cal: dict, min_coverage: float = 0.8) -> bool:
    """The model read evenly spaced frame numbers; a complete list has about
    clip_frames / gap of them. A shorter list means it stopped listing early, which
    would inflate tokens per frame, so the calibration isn't used."""
    gap, seen, total = cal.get("median_gap_frames"), cal.get("frames_read") or 0, cal.get("clip_frames") or 0
    if not gap or seen < 3 or not total:
        return False
    return seen * gap / total >= min_coverage


def density_limits(d: Optional[dict], model: str, cal: Optional[dict] = None) -> dict:
    """Video tokens per minute (measured) and frames per minute.

    Frames come from the frame-counter calibration: the model lists the frame
    numbers it saw in a 30 s clip, giving the real tokens per frame, which converts
    the reference clip's tokens into frames. Without a calibration, only Gemini gets
    an estimate (its 66 tokens/frame is documented and was confirmed by calibration)."""
    if not d:
        return {"video_tokens_per_min": {"status": "not_tested"}}
    if d.get("visual_tokens_per_min") is not None:
        ok = [x for x in d.get("runs", []) if x.get("visual_tokens_per_min") is not None]
        tok = {"status": "measured", "value": round(d["visual_tokens_per_min"]),
               "video": ", ".join(d.get("video_used", [])), "repeats": len(ok), "spread_pct": d.get("spread_pct")}
        if any("360p" in v for v in d.get("video_used", [])):
            tok["note"] = "The 40 MB original was rejected; measured on a 360p copy of the same clip."
        out = {"video_tokens_per_min": tok}
        tpf = (cal or {}).get("tokens_per_frame")
        if tpf and (cal or {}).get("frames_read") and calibration_complete(cal):
            fpm = d["visual_tokens_per_min"] / tpf
            dur_min = (ok[0].get("duration_s") or 308) / 60
            short_fpm = cal["frames_read"] / cal["clip_seconds"] * 60
            out["video_frames_per_min"] = {
                "status": "measured", "value": round(fpm, 1), "tokens_per_frame": tpf,
                "frames_per_video": round(fpm * dur_min), "video_minutes": round(dur_min, 2),
                "frames_per_min_short_clip": round(short_fpm, 1),
                # Far fewer frames per minute on the long clip than on the 30 s one:
                # the provider caps frames per video instead of sampling at a fixed rate.
                "capped": fpm < 0.6 * short_fpm,
                "calibration_reply": scrub(cal.get("reply"), 300),
            }
        elif "gemini" in model.lower() and d.get("est_frames_per_min") is not None:
            out["video_frames_per_min"] = {"status": "estimated", "value": d["est_frames_per_min"],
                                           "tokens_per_frame_assumed": 66}
        return out
    fail = next((x for x in d.get("runs", []) if not x.get("ok") and x.get("kind")), None)
    if fail is None:
        return {"video_tokens_per_min": {"status": "inconclusive", "reason": (d.get("error") or "no result")[:80]}}
    kind = fail["kind"]
    status = "inconclusive" if kind in SOFT_KINDS else "rejected" if kind in SIZE_KINDS else "unsupported"
    return {"video_tokens_per_min": {"status": status, "failure": kind, "evidence": scrub(fail.get("message"))}}


def group_twins(raw: list[dict]) -> list[dict]:
    """One target per (endpoint, model, pin). The OpenRouter sweep re-measures pins
    the main run already has (e.g. alibaba); its results fill dimensions the main
    run lacks. A target that only exists in the sweep is not featured."""
    def weak(x: Any) -> bool:
        return not isinstance(x, dict) or bool(x.get("error")) or bool(x.get("inconclusive"))

    groups: dict[tuple, dict] = {}
    for t in raw:
        is_sweep = " · openrouter · " in t["name"]
        k = (t["base_url"].rstrip("/").lower(), core_key(t["model"]), t.get("pin"))
        g = groups.get(k)
        if g is None:
            groups[k] = {**t, "results": dict(t["results"]), "sweep_only": is_sweep}
            continue
        main_first = g["sweep_only"] and not is_sweep
        if main_first:
            g["name"], g["sweep_only"] = t["name"], False
        for dim, res in t["results"].items():
            cur = g["results"].get(dim)
            if cur is None or (weak(cur) and not weak(res)):
                g["results"][dim] = res
            elif not weak(res) and not weak(cur):
                # Both conclusive: keep the tighter bracket (a search cut short
                # by an outage ends wider), and the main run's on a tie.
                w_new, w_cur = bracket(res), bracket(cur)
                if w_new < w_cur or (w_new == w_cur and main_first):
                    g["results"][dim] = res
    return list(groups.values())


def bracket(res: Any) -> float:
    """Width of a search's pass/fail bracket; smaller is more precise."""
    if not isinstance(res, dict) or res.get("max_ok") is None or res.get("fail_at") is None:
        return float("inf")
    return float(res["fail_at"]) - float(res["max_ok"])



def apply_observations(entries: list[dict], names: dict[str, str]) -> None:
    if not OBSERVATIONS_FILE.exists():
        return
    by_id = {e["id"]: e for e in entries}
    for ob in json.loads(OBSERVATIONS_FILE.read_text()):
        e = by_id.get(names.get(ob["target"], ""))
        if e is None:
            continue
        if ob.get("limit"):
            cur = e["limits"].get(ob["limit"], {"status": "not_tested"})
            if cur["status"] in ("not_tested", "inconclusive") or ob.get("override"):
                e["limits"][ob["limit"]] = {"status": "stated", "value": ob["value"], "evidence": ob["evidence"],
                                            "source": ob.get("source", "agent session")}
        if ob.get("note"):
            e["notes"].append(ob["note"])


def diff(old: list[dict], new: list[dict], on: str) -> list[dict]:
    """Changes between two datasets: added / removed entries and changed limit values."""
    o, n = {e["id"]: e for e in old}, {e["id"]: e for e in new}
    out = []
    for eid in sorted(n.keys() - o.keys()):
        out.append({"date": on, "id": eid, "change": "added"})
    for eid in sorted(o.keys() - n.keys()):
        out.append({"date": on, "id": eid, "change": "removed"})
    for eid in sorted(n.keys() & o.keys()):
        for k in sorted(set(n[eid]["limits"]) | set(o[eid]["limits"])):
            a, b = o[eid]["limits"].get(k, {}), n[eid]["limits"].get(k, {})
            fa = (a.get("status"), a.get("value"), a.get("stated"))
            fb = (b.get("status"), b.get("value"), b.get("stated"))
            if fa != fb:
                out.append({"date": on, "id": eid, "change": "limit", "limit": k,
                            "from": {k2: a.get(k2) for k2 in ("status", "value", "stated") if a.get(k2) is not None},
                            "to": {k2: b.get(k2) for k2 in ("status", "value", "stated") if b.get(k2) is not None}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, type=Path, help="probe.json from a probe run (or a merged run)")
    ap.add_argument("--catalog", required=True, type=Path, help="model catalog (OpenRouter /models format)")
    ap.add_argument("--date", default=date.today().isoformat(), help="measurement date (YYYY-MM-DD)")
    args = ap.parse_args()

    raw = json.loads(args.probe.read_text())
    catalog = json.loads(args.catalog.read_text()).get("data", [])
    overrides = json.loads((DATA / "models.json").read_text())["models"] if (DATA / "models.json").exists() else []
    entries, names = [], {}
    for t in group_twins(raw):
        e = entry_from_target(t, catalog, overrides, args.date)
        if e:
            entries.append(e)
            names[t["name"]] = e["id"]
    apply_observations(entries, names)
    entries.sort(key=lambda e: (e["model"]["name"].lower(), e["provider"]["name"].lower()))

    dataset = {"$schema": SCHEMA, "version": 1, "probe_version": PROBE_VERSION, "updated": args.date,
               "limits": {k: {"label": lbl, "unit": unit} for k, (_d, unit, lbl) in LIMITS.items()}
               | {"tokens_per_frame_768x432": {"label": "Tokens per 768×432 frame", "unit": "tokens"},
                  "video_tokens_per_min": {"label": "Video tokens per minute", "unit": "tokens"},
                  "video_frames_per_min": {"label": "Video frames per minute (estimated)", "unit": "count"}},
               "entries": entries}

    snaps = sorted(p for p in (DATA / "snapshots").glob("*.json") if p.stem != args.date)
    prev = json.loads(snaps[-1].read_text())["entries"] if snaps else []
    changes_path = DATA / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else []
    changes = [c for c in changes if c["date"] != args.date] + diff(prev, entries, args.date)

    DATA.mkdir(exist_ok=True)
    (DATA / "capabilities.json").write_text(json.dumps(dataset, indent=1, ensure_ascii=False) + "\n")
    (DATA / "snapshots" / f"{args.date}.json").write_text(json.dumps(dataset, indent=1, ensure_ascii=False) + "\n")
    changes_path.write_text(json.dumps(changes, indent=1, ensure_ascii=False) + "\n")

    leaks = re.findall(r"https?://(?!vlm\.fyi)|X-Amz-|user_[A-Za-z0-9]{10,}|sk-or-|tgp_v1|fw_[A-Za-z0-9]{8}|wk-[A-Za-z0-9]{6}|modal\.direct|--ep-",
                       (DATA / "capabilities.json").read_text())
    print(f"{len(entries)} entries · {len(changes)} change records · leak check: {'CLEAN' if not leaks else leaks[:5]}")
    if leaks:
        sys.exit(1)


if __name__ == "__main__":
    main()
