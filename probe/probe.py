"""vlm.fyi limit probe: measure what a provider + model accepts in one request.

Each request asks for "Reply with OK." (max_tokens 16) and carries synthetic media
built to stress ONE limit at a time, so limits don't mask each other:

  image_count     tiny 64x64 solid JPEGs       -> how many images per request
  inline_bytes    8 tiny JPEGs padded with      -> request-body bytes (no pixels)
                  JPEG comment segments
  url_bytes       the same, sent as S3 URLs     -> bytes the provider downloads
  video_bytes     a 5s clip padded with an MP4  -> video file size (no content)
                  `free` box, sent as video_url
  frames_768      768x432 solid frames          -> how many real-size frames fit
  video_density   a real reference video        -> visual tokens (and ≈ frames)
                  sent as video_url                the provider actually feeds the
                                                   model per minute of video

Limits are found by doubling until a failure, then bisecting; the provider's error
text is classified (PATTERNS) and kept. It also checks which request fields are
accepted and whether tool-message ordering is enforced.

Every request and response is recorded in requests.jsonl (headers, request ids,
status, body or streamed chunks, timings; base64 media replaced by sha256 + size),
so a result can be reproduced and shown to the provider.

    python probe.py --targets targets.toml                       # all targets
    python probe.py --targets targets_sweep.toml --only qwen3.8  # one model, every OpenRouter upstream
    python probe.py --merge                                      # combine runs into results/probe-merged/

Keys: .env.local next to this file (names referenced by api_key_env in the TOML).
URL/video probes need S3: S3_ENDPOINT, S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextvars
import hashlib
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse, urlunparse

import aiohttp
from dotenv import dotenv_values, load_dotenv
from PIL import Image
from rich.console import Console

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
console = Console() if sys.stdout.isatty() else Console(width=180)
MB = 1_000_000
PROBE_VERSION = "2"
DIMENSIONS = ("params", "tool_order", "video_bytes", "video_density", "video_frames", "image_count", "inline_bytes", "url_bytes", "frames_768")

# Reference videos for the density probe (the same clips as the August E06 sweep).
REF_VIDEOS = {
    "short": {"name": "baker_5min", "s3_key": "videos/vid_957c0817d0d644da.mp4", "duration_s": 308},
    "long": {"name": "accident_40min", "s3_key": "videos/vid_070ba8d480d17d86.mp4", "duration_s": 2417},
}
# Size-limited providers reject the 40 MB original (Gemini on AI Studio takes ~12 MB);
# they get the same clip re-encoded to 360p (< 10 MB). Results say which file was used.
SMALL_VARIANT = {"short": {"name": "baker_5min_360p", "s3_key": "probe/ref/baker_5min_360p.mp4", "duration_s": 308,
                           "from": "short", "vf": "scale=-2:360", "bitrate": "220k"}}
# Some providers decode only H.264/H.265; the reference clip is VP9. Same clip, same
# resolution, re-encoded to H.264 so tokens per minute stay comparable.
H264_VARIANT = {"short": {"name": "baker_5min_h264", "s3_key": "probe/ref/baker_5min_h264.mp4", "duration_s": 308,
                          "from": "short", "vf": None, "crf": "23"}}
# Duration-capped providers (Baseten: 300 s) get the first 4 minutes of the same clip.
TRIM_VARIANT = {"short": {"name": "baker_4min_h264", "s3_key": "probe/ref/baker_240s_h264.mp4", "duration_s": 240,
                          "from": "short", "vf": None, "crf": "23", "trim": 240}}
# Providers that feed every frame at full resolution (Fireworks: 2 fps at ~2,700 tokens per
# 1080p frame) can't fit 5 minutes in a 1M context: measure on the first minute instead.
MINUTE_VARIANT = {"short": {"name": "baker_1min_h264", "s3_key": "probe/ref/baker_60s_h264.mp4", "duration_s": 60,
                            "from": "short", "vf": None, "crf": "23", "trim": 60}}
SIZE_REJECTIONS = {"invalid_media", "media_bytes", "request_bytes", "download_bytes", "decoded_bytes"}

# Frame-counter calibration: the same resolution and frame rate as the reference
# clip it calibrates (providers resize by resolution, so tokens per frame must be
# measured at the same size). Every frame shows its own number; the model lists
# the numbers it saw, which gives the frames it actually received.
CALIBRATION = {
    "short": {"name": "counter_1080p_30s", "s3_key": "probe/ref/counter_1080p_30s.mp4", "w": 1920, "h": 1080, "fps": 30,
              "seconds": 30, "font": 520},
    "short_360p": {"name": "counter_360p_30s", "s3_key": "probe/ref/counter_360p_30s.mp4", "w": 640, "h": 360, "fps": 30,
                   "seconds": 30, "font": 180},
}
COUNTER_PROMPT = ("Each frame of this video shows one large number. List every distinct number you can see, in order, "
                  "separated by spaces. Output only the numbers.")
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                   "/Library/Fonts/Arial Bold.ttf"]
DENSITY_PROMPT = "Describe the scene in one sentence."

# Prompt tokens per *video* frame, by model family, for the ≈ frames estimate.
# Qwen3-VL: 32x32-px visual tokens (~338 for a 768x432 frame) with a 2x temporal
# merge -> ~169 per frame. Gemini: ~66 per frame at its default (low) media
# resolution. Others: unknown, so only visual tokens are reported.
TOKENS_PER_VIDEO_FRAME = {"qwen": 169.0, "gemini": 66.0}


DENSITY_TIMEOUT = 300
DENSITY_START: Optional[str] = None   # start density on a variant (e.g. "minute") instead of the full clip


def args_density_timeout() -> int:
    return DENSITY_TIMEOUT


def tokens_per_video_frame(model: str) -> Optional[float]:
    m = model.lower()
    return next((v for k, v in TOKENS_PER_VIDEO_FRAME.items() if k in m), None)


def service_of(base_url: str) -> str:
    b = base_url.lower()
    for host, svc in (("openrouter", "openrouter"), ("fireworks", "fireworks"), ("together", "together"),
                      ("jarvislabs", "jarvislabs"), ("modal.direct", "modal"), ("modal.run", "modal"),
                      ("moonshot.ai", "moonshot")):
        if host in b:
            return svc
    return "self_hosted"


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------

# (kind, regex). First match wins; the first numeric group, if any, is the limit
# the provider states. Seeded from errors seen in the provider matrix runs.
PATTERNS: list[tuple[str, str]] = [
    ("no_route", r"No allowed providers|No endpoints found"),
    # Per-image size limits (stated in MB): Fireworks "download size exceeded maximum size
    # limit of 10.0 MB", Baseten "Image too large: 10.1 MB (limit 10 MB)".
    ("media_bytes", r"Video \S+ download size exceeded maximum size limit of (\d+)(?:\.\d+)? ?MB"),
    ("image_bytes", r"download size exceeded maximum size limit of (\d+)(?:\.\d+)? ?MB|Image too large: [\d.]+ MB \(limit (\d+) MB\)"),
    ("video_duration", r"Video too long: [\d.]+ ?s \(limit (\d+) ?s\)"),
    ("media_bytes", r"Video too large: [\d.]+ MB \(limit (\d+) MB\)"),
    ("download_bytes", r"MAX_MEDIA_DOWNLOAD_SIZE_MB=(\d+)"),
    # The provider gave up fetching our media URL in time: about the fetch, not a size limit.
    ("timeout", r"The operation was aborted|\"error_type\": ?\"timeout\""),
    ("download_timeout", r"Timed out while downloading media|[Tt]imed? ?out (?:while )?(?:fetching|downloading)"),
    # The provider could not fetch our media URL: about the fetch, not a limit.
    ("download_failed", r"could not be reached|Failed to download one or more|image_url_unreachable|error while fetching media"),
    # Codec, not video support: these decoders take H.264/H.265 only.
    ("codec", r"has no decoder|must be H\.264|decode only H\.264|[Uu]nsupported (?:video )?codec"),
    ("decoded_bytes", r"decoded video RGB payload|MAX_DECODED_BYTES"),
    # The provider's own server is broken (e.g. Fireworks: "Cannot process videos: libGL.so.1:
    # cannot open shared object file"): says nothing about what the API accepts.
    ("provider_error", r"cannot open shared object file|Internal Server Error|Model request failed|\"upstream_error\""),
    ("auth_or_billing", r"Credit limit|Unauthorized|invalid api key|Invalid API key|insufficient.{0,20}(?:credit|quota|balance)"),
    ("image_count", r"limit the number of images per conversation to (\d+)|max number of data-uri items\s*:?\s*(\d+)"
                    r"|[Tt]oo many images: \d+ \(max (\d+)\)|[Tt]oo many images in request: \d+ > (\d+)"
                    r"|[Aa]t most (\d+) image|too many images|images? per (?:request|prompt|message)"),
    ("download_bytes", r"[Dd]ownloaded .{0,40}cannot exceed (\d+)\s*MB"),
    ("media_bytes", r"media exceeds size limit: max (\d+) bytes|too large: \d+ exceeds limit (\d+)|file (?:is )?too large|exceeds the maximum (?:file )?size"),
    ("decoded_bytes", r"DECODE_BYTES=(\d+)|single-message ceiling of (\d+) bytes"),
    ("decoded_bytes", r"[Mm]ultimodal payload|decoded media exceeds"),
    ("request_bytes", r"Request body exceeds|request_too_large|[Pp]ayload [Tt]oo [Ll]arge|Request Entity Too Large"),
    ("body_timeout", r"[Rr]equest body timed out|request_timeout"),
    ("context_length", r"context length|maximum context|context_length_exceeded|too many tokens|prompt is too long"
                       r"|input token count exceeds|maximum number of tokens allowed"
                       r"|exceeds the model's|input tokens? exceed|longer than the model"),
    # Media by URL refused, not media itself (Moonshot's API takes base64 or uploaded files only).
    ("url_unsupported", r"[Uu]nsupported (?:image|video) url|URL must be from one of the allowed domains"),
    ("no_video", r"number of videos per conversation to 0|Video count \d+ exceeds limit 0|supports image input only|[Vv]ideo.{0,30}not supported|does not support video|[Uu]nsupported .{0,20}video"),
    ("strict_params", r"Extra inputs are not permitted|Unrecognized request argument|[Uu]nknown (?:field|parameter)"),
    ("tool_order", r"tool_call_id|parallel tool|tool messages responding|parallel group"),
    ("invalid_media", r"Invalid image or video input|rejected the request parameters|INVALID_ARGUMENT|[Ii]nvalid (?:image|video)"),
]
_SIGNED = re.compile(r"(https?://[^\s\"'\\?]+)\?[^\s\"'\\]*")


def short_msg(text: str, limit: int = 1500) -> str:
    """An error message to keep: signed URLs lose their query string first, so a long
    presigned URL can't push the provider's actual reason past the cut."""
    return _SIGNED.sub(r"\1?…", text or "")[:limit]


# A video request refused for reasons that may only concern the URL: retried inline.
INLINE_RETRY = {"url_unsupported", "no_video", "no_route"}
INCONCLUSIVE = {"transient", "timeout", "auth_or_billing", "no_route", "credit_guard", "download_timeout",
                "provider_error", "download_failed", "codec"}


# OpenRouter routing filters that answer a capability question: if one removed the
# pinned upstream, the upstream lacks that capability (not a routing hiccup).
_CAPABILITY_FILTERS = (("Input Video Support", "no_video"), ("Parameters", "strict_params"),
                       ("Context Length", "context_length"))


def classify_for(t: "Target", status: int, message: str) -> tuple[str, Optional[int]]:
    """classify(), plus OpenRouter no-route errors that are definite answers."""
    kind, stated = classify(status, message)
    if kind != "no_route" or t.service != "openrouter":
        return kind, stated
    if "No endpoints found that support input video" in message:
        return "no_video", None
    for name, answer in _CAPABILITY_FILTERS:
        m = re.search(rf"Filter by {name} removed ([^;\"]*)", message)
        if t.pin and m and t.pin in [x.strip().rstrip(".") for x in m.group(1).split(",")]:
            return answer, None
    return kind, stated


def classify(status: int, message: str) -> tuple[str, Optional[int]]:
    for kind, pat in PATTERNS:
        m = re.search(pat, message)
        if m:
            nums = [g for g in m.groups() if g and g.isdigit()]
            return kind, int(nums[0]) if nums else None
    if status == 413:
        return "request_bytes", None
    if status in (401, 402, 403):
        return "auth_or_billing", None
    return "other", None


# --------------------------------------------------------------------------
# Request log: every request and response, for reconciliation and as evidence
# --------------------------------------------------------------------------

CTX: contextvars.ContextVar[dict] = contextvars.ContextVar("ctx", default={})


class RequestLog:
    def __init__(self, path: Path) -> None:
        self.f = path.open("a", buffering=1)

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, default=str) + "\n")


LOG: Optional[RequestLog] = None
GUARD: Optional["CreditGuard"] = None
# Waits between attempts after a 429/5xx/connection error. Rate limits on shared
# upstreams can last minutes; giving up early turns a busy provider into a false "no".
RETRY_WAITS = (5, 20, 60, 90, 120)
_PACE: dict[str, list] = {}          # api key -> [lock, next allowed send time]
_AFTER = re.compile(r"(?:after|in)\s+(\d+(?:\.\d+)?)\s*s", re.I)


async def pace(t: "Target") -> None:
    """Keep a key under its requests-per-minute limit (targets that share a key share the budget)."""
    if not t.rpm:
        return
    lock, _ = _PACE.setdefault(t.key, [asyncio.Lock(), 0.0])
    async with lock:
        wait = _PACE[t.key][1] - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        _PACE[t.key][1] = time.time() + 60.0 / t.rpm


def known(r: "Result") -> Optional[bool]:
    """A yes/no check's answer: None when the request never got a real verdict
    (rate limited, timed out, unroutable) rather than a false "no"."""
    return True if r.ok else None if r.kind in INCONCLUSIVE else False
_DATA_URI = re.compile(r"^data:([\w/+.-]+);base64,(.*)$", re.S)


def redact(obj: Any) -> Any:
    """The request body as sent, minus bulk and secrets: base64 media becomes
    {mime, bytes, sha256}; signed URLs lose their query string (the path stays)."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 6 and all(o == obj[0] for o in obj):
            return [redact(obj[0]), f"... ×{len(obj)} identical"]
        return [redact(o) for o in obj]
    if isinstance(obj, str):
        m = _DATA_URI.match(obj)
        if m:
            raw = base64.b64decode(m.group(2))
            return {"data_uri": {"mime": m.group(1), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
        if obj.startswith("http") and "X-Amz-" in obj:
            u = urlparse(obj)
            return urlunparse(u._replace(query="<signed>"))
    return obj


def keep_headers(h: Any) -> dict:
    return {k: v for k, v in h.items() if k.lower() not in ("set-cookie",)}


# --------------------------------------------------------------------------
# Synthetic media and S3
# --------------------------------------------------------------------------

def solid_jpeg(w: int, h: int, color=(90, 140, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def pad_jpeg(jpeg: bytes, total: int) -> bytes:
    """Grow a JPEG to `total` bytes with comment (COM) segments right after SOI.
    Decoders skip COM, so bytes grow while pixels stay 64x64."""
    need = total - len(jpeg)
    segs = []
    while need > 4:
        n = min(need - 4, 65533)
        segs.append(b"\xff\xfe" + struct.pack(">H", n + 2) + b"\0" * n)
        need -= n + 4
    return jpeg[:2] + b"".join(segs) + jpeg[2:]


def pad_mp4(mp4: bytes, total: int) -> bytes:
    """Grow an MP4 to `total` bytes with a trailing `free` box (ignored by demuxers)."""
    size = max(8, total - len(mp4))
    return mp4 + struct.pack(">I", size) + b"free" + b"\0" * (size - 8)


def data_uri(b: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


class S3:
    def __init__(self) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = os.environ["S3_BUCKET"]
        self.client = boto3.client(
            "s3", endpoint_url=os.environ.get("S3_ENDPOINT") or None,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "auto"),
            config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
        )

    def upload(self, path: str, key: str, content_type: str) -> None:
        self.client.upload_file(path, self.bucket, key, ExtraArgs={"ContentType": content_type})

    def presign(self, key: str, expires: int = 6 * 3600) -> str:
        return self.client.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires)


def ref_spec(which: str, variant: Optional[str] = None) -> dict:
    """The reference clip, its 360p copy ("small") or its H.264 copy ("h264")."""
    return {"small": SMALL_VARIANT, "h264": H264_VARIANT, "trim": TRIM_VARIANT,
            "minute": MINUTE_VARIANT}.get(variant or "", REF_VIDEOS)[which]


class Fixtures:
    """Shared synthetic media; S3 uploads are cached by content key across targets."""

    def __init__(self) -> None:
        self.tiny = solid_jpeg(64, 64)
        self.frame = solid_jpeg(768, 432)
        self._urls: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._s3: Optional[S3] = None
        self._mp4: Optional[bytes] = None

    @property
    def s3(self) -> S3:
        if self._s3 is None:
            self._s3 = S3()
        return self._s3

    def base_mp4(self) -> bytes:
        if self._mp4 is None:
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "base.mp4")
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
                self._mp4 = Path(out).read_bytes()
        return self._mp4

    async def url(self, key: str, make: Callable[[], bytes], content_type: str) -> str:
        if key in self._urls:
            return self._urls[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self._urls:
                data = make()

                def _upload() -> str:
                    with tempfile.NamedTemporaryFile(delete=False) as f:
                        f.write(data)
                    try:
                        self.s3.upload(f.name, f"probe/{key}", content_type)
                    finally:
                        os.unlink(f.name)
                    return self.s3.presign(f"probe/{key}")

                self._urls[key] = await asyncio.to_thread(_upload)
        return self._urls[key]

    def counter_url(self, which: str) -> str:
        spec = CALIBRATION[which]
        k = f"ref:{spec['s3_key']}"
        if k not in self._urls:
            try:
                self.s3.client.head_object(Bucket=self.s3.bucket, Key=spec["s3_key"])
            except Exception:  # noqa: BLE001 - build it once
                font = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
                fontarg = f"fontfile={font}:" if font else ""
                with tempfile.TemporaryDirectory() as d:
                    out = os.path.join(d, "counter.mp4")
                    vf = (f"drawtext={fontarg}text='%{{eif\\:n\\:d\\:4}}':fontsize={spec['font']}:fontcolor=white:"
                          "x=(w-text_w)/2:y=(h-text_h)/2")
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                                    f"color=c=black:s={spec['w']}x{spec['h']}:r={spec['fps']}:d={spec['seconds']}",
                                    "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
                                    "-movflags", "+faststart", out], check=True)
                    self.s3.upload(out, spec["s3_key"], "video/mp4")
            self._urls[k] = self.s3.presign(spec["s3_key"])
        return self._urls[k]

    def data_uri(self, s3_key: str) -> str:
        """A reference video as a base64 data URL, for providers that refuse video URLs."""
        k = f"data:{s3_key}"
        if k not in self._urls:
            body = self.s3.client.get_object(Bucket=self.s3.bucket, Key=s3_key)["Body"].read()
            self._urls[k] = f"data:video/mp4;base64,{base64.b64encode(body).decode()}"
        return self._urls[k]

    def ref_video(self, t: "Target", which: str, variant: Optional[str] = None, inline: Optional[bool] = None) -> str:
        url = self.ref_video_url(which, variant)    # also builds the re-encode if needed
        inline = t.video_inline if inline is None else inline
        return self.data_uri(ref_spec(which, variant)["s3_key"]) if inline else url

    def ref_video_url(self, which: str, variant: Optional[str] = None) -> str:
        ref = ref_spec(which, variant)
        k = f"ref:{ref['s3_key']}"
        if k not in self._urls:
            if variant:
                self._ensure_small(ref)
            self._urls[k] = self.s3.presign(ref["s3_key"])
        return self._urls[k]

    def _ensure_small(self, ref: dict) -> None:
        """Build the small re-encode of a reference video once and keep it in S3."""
        try:
            self.s3.client.head_object(Bucket=self.s3.bucket, Key=ref["s3_key"])
            return
        except Exception:  # noqa: BLE001 - not there yet
            pass
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "src.mp4"), os.path.join(d, "small.mp4")
            self.s3.client.download_file(self.s3.bucket, REF_VIDEOS[ref["from"]]["s3_key"], src)
            rate = (["-crf", ref["crf"]] if ref.get("crf") else
                    ["-b:v", ref["bitrate"], "-maxrate", ref["bitrate"], "-bufsize", "500k"])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *(["-t", str(ref["trim"])] if ref.get("trim") else []),
                            *(["-vf", ref["vf"]] if ref.get("vf") else []),
                            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", *rate, "-an",
                            "-movflags", "+faststart", out], check=True)
            self.s3.upload(out, ref["s3_key"], "video/mp4")


# --------------------------------------------------------------------------
# One request
# --------------------------------------------------------------------------

@dataclass
class Result:
    ok: bool
    status: int = 0
    kind: str = ""
    stated: Optional[int] = None
    message: str = ""
    seconds: float = 0.0
    body_bytes: int = 0
    prompt_tokens: Optional[int] = None
    request_id: Optional[str] = None
    text: Optional[str] = None


@dataclass
class Target:
    name: str
    base_url: str
    model: str
    key: str
    pin: Optional[str] = None
    dims: Optional[set[str]] = None
    service: str = ""
    rpm: Optional[float] = None       # account rate limit: space this key's requests out
    video_inline: bool = False        # provider refuses video URLs: send reference videos as base64
    accepts: dict[str, bool] = field(default_factory=dict)
    out: dict[str, Any] = field(default_factory=dict)
    requests: int = 0
    prompt_tokens: int = 0

    def want(self, dim: str, skip: set[str]) -> bool:
        return dim not in skip and (self.dims is None or dim in self.dims)


def img(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def image_messages(urls: list[str], per_msg: int = 50) -> list[dict]:
    """Spread images over several user messages, like an agent's context + clip messages."""
    msgs = []
    for i in range(0, len(urls), per_msg):
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"Images {i + 1}-{min(i + per_msg, len(urls))}."}]
                     + [img(u) for u in urls[i:i + per_msg]]})
    msgs.append({"role": "user", "content": "Reply with OK."})
    return msgs


class _Transient(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


def _request_id(headers: dict, first_chunk: Optional[dict]) -> Optional[str]:
    for h in ("x-generation-id", "x-request-id", "request-id", "x-amzn-requestid", "fireworks-request-id", "x-together-request-id", "cf-ray"):
        for k, v in headers.items():
            if k.lower() == h:
                return v
    return (first_chunk or {}).get("id") if isinstance(first_chunk, dict) else None


async def call(http: aiohttp.ClientSession, t: Target, messages: list[dict], extra: Optional[dict] = None,
               timeout: int = 300, stream: bool = True, max_tokens: int = 16) -> Result:
    body: dict[str, Any] = {"model": t.model, "messages": messages, "max_tokens": max_tokens, "stream": stream}
    if stream and t.accepts.get("stream_options"):
        body["stream_options"] = {"include_usage": True}
    if t.pin and t.service == "openrouter":
        body["provider"] = {"order": [t.pin], "allow_fallbacks": False}
    body.update(extra or {})
    data = json.dumps(body).encode()
    url = t.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {t.key}", "Content-Type": "application/json",
               "Accept": "text/event-stream" if stream else "application/json",
               "Accept-Encoding": "gzip, deflate"}   # some gateways send brotli the client can't always decode
    ctx = CTX.get()
    req_summary = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "body": redact(body)}

    last = Result(False, 0, "transient", None, "no attempt")
    if t.service == "openrouter" and GUARD is not None and not await GUARD.ok():
        # Checked before every request: parallel targets all pass a start-of-target
        # check before any money is spent, so that alone can overshoot badly.
        return Result(False, 0, "credit_guard", None, "OpenRouter balance below the floor; request not sent")
    for attempt in range(len(RETRY_WAITS) + 1):
        await pace(t)
        t0, ttfb = time.time(), None
        t.requests += 1
        rec: dict[str, Any] = {"id": str(uuid.uuid4()), "ts": datetime.now(timezone.utc).isoformat(), "target": t.name,
                               "model": t.model, "pin": t.pin, "endpoint": urlparse(url).netloc, "path": urlparse(url).path,
                               **ctx, "attempt": attempt + 1, "request": req_summary}
        chunks: list[Any] = []
        resp_headers: dict = {}
        status = 0
        try:
            async with http.post(url, data=data, headers=headers,
                                 timeout=aiohttp.ClientTimeout(total=timeout, sock_read=timeout)) as resp:
                status, resp_headers = resp.status, keep_headers(resp.headers)
                if resp.status != 200 or not stream:
                    text = await resp.text()
                    ttfb = time.time() - t0
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = None
                    rec["response"] = {"status": status, "headers": resp_headers, "body": parsed if parsed is not None else text[:20000]}
                    if resp.status == 429 or resp.status >= 500:
                        raise _Transient(resp.status, text[:1500])
                    if resp.status != 200:
                        kind, stated = classify_for(t, resp.status, text)
                        r = Result(False, resp.status, kind, stated, short_msg(text), time.time() - t0, len(data),
                                   request_id=_request_id(resp_headers, parsed if isinstance(parsed, dict) else None))
                        return _log(rec, r, ttfb)
                    if isinstance(parsed, dict) and parsed.get("error") and not parsed.get("choices"):
                        msg = json.dumps(parsed["error"])
                        kind, stated = classify_for(t, 200, msg)
                        return _log(rec, Result(False, 200, kind, stated, msg[:1500], time.time() - t0, len(data),
                                                request_id=_request_id(resp_headers, parsed)), ttfb)
                    pt = ((parsed or {}).get("usage") or {}).get("prompt_tokens")
                    if pt:
                        t.prompt_tokens += int(pt)
                    msg0 = ((((parsed or {}).get("choices") or [{}])[0]).get("message") or {})
                    return _log(rec, Result(True, 200, seconds=time.time() - t0, body_bytes=len(data),
                                            prompt_tokens=int(pt) if pt else None,
                                            request_id=_request_id(resp_headers, parsed),
                                            text=msg0.get("content") if isinstance(msg0.get("content"), str) else None), ttfb)

                prompt_tokens = None
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    if ttfb is None:
                        ttfb = time.time() - t0
                    if line == "data: [DONE]":
                        chunks.append("[DONE]")
                        continue
                    try:
                        chunk = json.loads(line[5:])
                    except json.JSONDecodeError:
                        chunks.append(line[:2000])
                        continue
                    if len(chunks) < 300:
                        chunks.append(chunk)
                    if isinstance(chunk, dict) and chunk.get("error") and not chunk.get("choices"):
                        rec["response"] = {"status": status, "headers": resp_headers, "chunks": chunks}
                        msg = json.dumps(chunk["error"])[:1500]
                        code = chunk["error"].get("code") if isinstance(chunk["error"], dict) else None
                        if isinstance(code, int) and (code == 429 or code >= 500):
                            raise _Transient(code, msg)
                        kind, stated = classify_for(t, code if isinstance(code, int) else 200, msg)
                        return _log(rec, Result(False, 200, kind, stated, msg, time.time() - t0, len(data),
                                                request_id=_request_id(resp_headers, chunk)), ttfb)
                    usage = chunk.get("usage") if isinstance(chunk, dict) else None
                    if isinstance(usage, dict) and usage.get("prompt_tokens"):
                        prompt_tokens = int(usage["prompt_tokens"])
                rec["response"] = {"status": status, "headers": resp_headers, "chunks": chunks}
                if prompt_tokens:
                    t.prompt_tokens += prompt_tokens
                first = next((c for c in chunks if isinstance(c, dict)), None)
                return _log(rec, Result(True, 200, seconds=time.time() - t0, body_bytes=len(data), prompt_tokens=prompt_tokens,
                                        request_id=_request_id(resp_headers, first)), ttfb)
        except _Transient as e:
            last = Result(False, e.status, "transient", None, e.message, time.time() - t0, len(data))
            rec.setdefault("response", {"status": status, "headers": resp_headers, "chunks": chunks})
            _log(rec, last, ttfb)
        except asyncio.TimeoutError:
            rec["response"] = {"status": status, "headers": resp_headers, "chunks": chunks}
            return _log(rec, Result(False, status, "timeout", None, f"no complete response within {timeout}s",
                                    time.time() - t0, len(data)), ttfb)
        except aiohttp.ClientError as e:
            last = Result(False, 0, "transient", None, f"{type(e).__name__}: {e}"[:500], time.time() - t0, len(data))
            rec["response"] = {"status": status, "headers": resp_headers, "error": f"{type(e).__name__}: {e}"}
            _log(rec, last, ttfb)
        if attempt == len(RETRY_WAITS):
            break
        hint = _AFTER.search(last.message or "")
        retry_after = (resp_headers or {}).get("Retry-After") or (resp_headers or {}).get("retry-after")
        wait = RETRY_WAITS[attempt]
        for v in (hint.group(1) if hint else None, retry_after):
            try:
                wait = max(wait, min(float(v) + 1, 300))
            except (TypeError, ValueError):
                pass
        await asyncio.sleep(wait)
    return last


def _log(rec: dict, r: Result, ttfb: Optional[float]) -> Result:
    rec["timing"] = {"ttfb_s": round(ttfb, 3) if ttfb is not None else None, "total_s": round(r.seconds, 3)}
    rec["result"] = {"ok": r.ok, "kind": r.kind, "stated": r.stated, "prompt_tokens": r.prompt_tokens, "request_id": r.request_id}
    if LOG:
        LOG.write(rec)
    return r


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

async def search(probe: Callable[[float], Awaitable[Result]], start: float, cap: float, *,
                 integer: bool, tol: float) -> dict:
    """Double from `start` until a failure (or `cap`), then bisect between the last
    pass and the first failure. Inconclusive failures (transient, timeout) stop it."""
    tried: list[dict] = []
    base_ctx = CTX.get()

    async def run(v: float) -> Result:
        v = int(round(v)) if integer else v
        token = CTX.set({**base_ctx, "value": v})
        try:
            r = await probe(v)
        finally:
            CTX.reset(token)
        tried.append({"value": v, "ok": r.ok, "kind": r.kind, "seconds": round(r.seconds, 1),
                      **({"prompt_tokens": r.prompt_tokens} if r.prompt_tokens else {}),
                      **({"request_id": r.request_id} if r.request_id else {})})
        return r

    def is_limit(r: Result, passed: bool) -> bool:
        # OpenRouter reports an over-limit request on a pinned upstream as "no
        # endpoints found" (nothing left to route to). After a smaller request passed,
        # that is the limit, not an outage (the August E07 finding).
        return r.kind not in INCONCLUSIVE or (r.kind == "no_route" and passed)

    lo, hi, fail = None, None, None
    v = start
    while True:
        r = await run(v)
        if not r.ok:
            if not is_limit(r, lo is not None):
                return {"max_ok": lo, "fail_at": None, "inconclusive": r.kind, "message": r.message, "tried": tried}
            hi, fail = (int(round(v)) if integer else v), r
            break
        lo = int(round(v)) if integer else v
        if v >= cap:
            return {"max_ok": lo, "fail_at": None, "note": "no failure up to the probe cap", "tried": tried}
        v = min(cap, v * 2)

    a, b = (lo or 0), hi
    while True:
        if integer and b - a <= max(1, int(b * tol)):
            break
        if not integer and (a > 0 and (b - a) / b <= tol or a == 0 and b <= start):
            break
        mid = (a + b) / 2 if a else b / 2
        if integer:
            mid = int(mid)
            if mid in (a, b) or mid < 1:
                break
        r = await run(mid)
        if r.ok:
            a = mid
        elif not is_limit(r, a > 0):
            break
        else:
            b, fail = mid, r
    return {"max_ok": a or None, "fail_at": b, "fail_kind": fail.kind, "stated_limit": fail.stated,
            "message": short_msg(fail.message, 600), "request_id": fail.request_id, "tried": tried}


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------

TEXT = [{"role": "user", "content": "Reply with OK."}]
NOOP_TOOL = [{"type": "function", "function": {"name": "noop", "description": "Does nothing.",
                                               "parameters": {"type": "object", "properties": {}}}}]
PARAMS = {
    "stream_options": {"stream_options": {"include_usage": True}},
    "usage": {"usage": {"include": True}},
    "reasoning": {"reasoning": {"enabled": True}},
    "response_format": {"response_format": {"type": "json_schema", "json_schema": {
        "name": "reply", "strict": True,
        "schema": {"type": "object", "properties": {"ok": {"type": "string"}}, "required": ["ok"],
                   "additionalProperties": False}}}},
    "tools": {"tools": NOOP_TOOL, "tool_choice": "auto"},
}


async def density(http: aiohttp.ClientSession, t: Target, fx: Fixtures, videos: list[str], repeats: int) -> dict:
    """Visual tokens the provider bills for a real reference video, per minute of video.

    prompt_tokens(video + prompt) - prompt_tokens(prompt alone) = the video's tokens.
    Exact, model-agnostic. The ≈ frames figure divides by a per-family tokens-per-
    frame constant and is an estimate (the provider's frame resolution is unknown)."""
    text_only = [{"role": "user", "content": [{"type": "text", "text": DENSITY_PROMPT}]}]
    CTX.set({**CTX.get(), "value": "text_baseline"})
    base = await call(http, t, text_only, stream=False, max_tokens=8, timeout=180)
    out: dict[str, Any] = {"prompt": DENSITY_PROMPT, "baseline_prompt_tokens": base.prompt_tokens, "runs": [],
                           **({"video_inline": True} if t.video_inline else {})}
    if not base.ok or not base.prompt_tokens:
        out["error"] = f"text baseline failed or reported no usage: {base.kind} {base.message[:200]}"
        return out
    per_frame = tokens_per_video_frame(t.model)
    inline = t.video_inline
    for which in videos:
        variant: Optional[str] = DENSITY_START if DENSITY_START and which in MINUTE_VARIANT else None
        ref = ref_spec(which, variant)
        tried = {variant}            # variants already sent, so the fallbacks below can't loop
        for rep in range(repeats):
            while True:
                msgs = [{"role": "user", "content": [{"type": "text", "text": DENSITY_PROMPT},
                                                     {"type": "video_url", "video_url": {
                                                         "url": await asyncio.to_thread(fx.ref_video, t, which, variant, inline)}}]}]
                CTX.set({**CTX.get(), "value": f"{ref['name']}#{rep + 1}{' inline' if inline else ''}"})
                r = await call(http, t, msgs, stream=False, max_tokens=8, timeout=args_density_timeout())
                if r.ok:
                    break
                if r.kind in INLINE_RETRY and not inline:
                    # The video URL was refused or filtered out (OpenRouter drops upstreams that
                    # can't fetch URLs): that's about URLs, so try the same video inline.
                    note, inline = "video URL refused; retrying inline (base64)", True
                elif r.kind in ("timeout", "context_length") and "minute" not in tried and which in MINUTE_VARIANT:
                    note, variant = "too many tokens or too slow for the full clip; retrying with its first minute", "minute"
                elif r.kind == "video_duration" and "trim" not in tried and which in TRIM_VARIANT:
                    note, variant = "longer than the provider's video length limit; retrying with the first 4 minutes", "trim"
                elif r.kind == "codec" and "h264" not in tried and which in H264_VARIANT:
                    note, variant = "the provider can't decode VP9; retrying with an H.264 copy", "h264"
                elif r.kind in SIZE_REJECTIONS and "small" not in tried and which in SMALL_VARIANT:
                    note, variant = "retrying with the 360p copy", "small"
                elif (r.kind in SIZE_REJECTIONS or r.kind == "video_duration") and "minute" not in tried and which in MINUTE_VARIANT:
                    # Still too large or too long: the last fallback is the clip's first minute.
                    note, variant = "still refused; retrying with the clip's first minute", "minute"
                else:
                    break
                tried.add(variant)
                out["runs"].append({"video": ref["name"], "repeat": rep + 1, "ok": False, "kind": r.kind,
                                    "message": r.message[:400], "request_id": r.request_id, "note": note})
                ref = ref_spec(which, variant)
            if inline:
                out["video_inline"] = True
            run: dict[str, Any] = {"video": ref["name"], "duration_s": ref["duration_s"], "repeat": rep + 1,
                                   "ok": r.ok, "latency_s": round(r.seconds, 1), "request_id": r.request_id}
            if r.ok and r.prompt_tokens:
                vis = max(0, r.prompt_tokens - base.prompt_tokens)
                run.update({"prompt_tokens": r.prompt_tokens, "visual_tokens": vis,
                            "visual_tokens_per_min": round(vis / (ref["duration_s"] / 60), 1)})
                if per_frame:
                    frames = vis / per_frame
                    run.update({"est_frames": round(frames), "est_frames_per_min": round(frames / (ref["duration_s"] / 60), 1),
                                "tokens_per_frame_assumed": per_frame})
            elif r.ok:
                run["error"] = "accepted, but reported no usage"
            else:
                run.update({"kind": r.kind, "message": r.message[:400]})
            out["runs"].append(run)
            if not r.ok:
                break
    ok = [x for x in out["runs"] if x.get("visual_tokens_per_min") is not None]
    if ok:
        out["video_used"] = sorted({x["video"] for x in ok})
        vals = [x["visual_tokens_per_min"] for x in ok]
        out["visual_tokens_per_min"] = round(sum(vals) / len(vals), 1)
        out["spread_pct"] = round(100 * (max(vals) - min(vals)) / max(1e-9, max(vals)), 1)
        fpm = [x["est_frames_per_min"] for x in ok if x.get("est_frames_per_min") is not None]
        if fpm:
            out["est_frames_per_min"] = round(sum(fpm) / len(fpm), 1)
    return out


def read_numbers(text: str, n_frames: int) -> list[int]:
    """Frame numbers in a reply, ignoring any <think> block; distinct, in range, sorted."""
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S)
    return sorted({int(x) for x in re.findall(r"(?<!\d)\d{1,4}(?!\d)", text) if int(x) < n_frames})


async def frame_calibration(http: aiohttp.ClientSession, t: Target, fx: Fixtures, which: str, density: Optional[dict]) -> dict:
    """Frames the model actually received from a frame-counter clip, and from that
    the real tokens per frame, which turns the density probe's tokens/min into frames/min."""
    spec = CALIBRATION[which]
    n_frames = spec["fps"] * spec["seconds"]
    CTX.set({**CTX.get(), "value": "text_baseline"})
    base = await call(http, t, [{"role": "user", "content": [{"type": "text", "text": COUNTER_PROMPT}]}],
                      stream=False, max_tokens=8, timeout=180)
    out: dict[str, Any] = {"clip": spec["name"], "clip_frames": n_frames, "clip_seconds": spec["seconds"],
                           "baseline_prompt_tokens": base.prompt_tokens}
    if not base.ok or not base.prompt_tokens:
        out["error"] = f"text baseline failed or reported no usage: {base.kind}"
        return out
    url = await asyncio.to_thread(fx.counter_url, which)
    if t.video_inline or (density or {}).get("video_inline"):
        url = await asyncio.to_thread(fx.data_uri, spec["s3_key"])
    CTX.set({**CTX.get(), "value": spec["name"]})
    msgs = [{"role": "user", "content": [{"type": "text", "text": COUNTER_PROMPT},
                                         {"type": "video_url", "video_url": {"url": url}}]}]
    r = await call(http, t, msgs, stream=False, max_tokens=6000, timeout=args_density_timeout())
    if r.ok and not (r.text or "").strip():
        # A reasoning model can spend the whole output budget thinking (GLM 5.3 Flash
        # did: 6,000 reasoning tokens, empty reply). Ask for less thinking and allow more
        # output; drop the effort field if the provider refuses it.
        for extra, label in (({"reasoning_effort": "low"}, "reasoning_effort=low, max_tokens=16000"),
                             ({}, "max_tokens=16000")):
            CTX.set({**CTX.get(), "value": f"{spec['name']} retry: {label}"})
            again = await call(http, t, msgs, extra, stream=False, max_tokens=16000, timeout=args_density_timeout())
            if again.ok:
                r = again
                out["retry"] = f"empty reply at max_tokens=6000; retried with {label}"
                break
    out.update({"ok": r.ok, "request_id": r.request_id, "latency_s": round(r.seconds, 1)})
    if not r.ok:
        out.update({"kind": r.kind, "message": r.message[:400]})
        return out
    nums = read_numbers(r.text or "", n_frames)
    out["reply"] = (r.text or "")[:3000]
    out["prompt_tokens"] = r.prompt_tokens
    if not nums or not r.prompt_tokens:
        out["error"] = "no frame numbers in the reply" if not nums else "no usage reported"
        return out
    gaps = [b - a for a, b in zip(nums, nums[1:])]
    gaps.sort()
    vis = max(0, r.prompt_tokens - base.prompt_tokens)
    seen = len(nums)
    out.update({
        "frames_read": seen, "first": nums[:5], "last": nums[-5:],
        "median_gap_frames": gaps[len(gaps) // 2] if gaps else None,
        "frames_per_min_on_clip": round(seen / spec["seconds"] * 60, 1),
        "visual_tokens": vis, "tokens_per_frame": round(vis / seen, 1),
    })
    if density and density.get("visual_tokens_per_min"):
        out["frames_per_min_calibrated"] = round(density["visual_tokens_per_min"] / out["tokens_per_frame"], 1)
    return out


async def probe_target(http: aiohttp.ClientSession, t: Target, fx: Fixtures, skip: set[str], log, args) -> None:
    def done(dim: str, res: Any) -> None:
        t.out[dim] = res
        log(t, dim, res)

    def stage(dim: str) -> None:
        CTX.set({"dimension": dim})

    stage("baseline")
    base = await call(http, t, TEXT)
    done("baseline", {"ok": base.ok, "kind": base.kind, "message": base.message[:300], "seconds": round(base.seconds, 1),
                      "request_id": base.request_id})
    if not base.ok:
        return

    if t.want("params", skip):
        stage("params")
        res = {}
        for name, extra in PARAMS.items():
            CTX.set({"dimension": "params", "value": name})
            r = await call(http, t, TEXT, extra)
            res[name] = {"ok": known(r), **({} if r.ok else {"kind": r.kind, "message": r.message[:200]})}
            t.accepts[name] = r.ok
        done("params", res)
    else:
        t.accepts["stream_options"] = True

    stage("image_input")
    r = await call(http, t, image_messages([data_uri(fx.tiny)]))
    t.accepts["image_inline"] = r.ok
    try:
        tiny_url = await fx.url("tiny.jpg", lambda: fx.tiny, "image/jpeg")
        url_res = await call(http, t, image_messages([tiny_url]))
    except Exception as e:  # noqa: BLE001 - no S3: URL probes are skipped
        url_res = Result(False, 0, "no_s3", None, str(e)[:200])
    t.accepts["image_url"] = url_res.ok
    done("image_input", {"inline": known(r), "url": known(url_res),
                         **({} if r.ok else {"inline_error": f"{r.kind}: {r.message[:200]}"}),
                         **({} if url_res.ok else {"url_error": f"{url_res.kind}: {url_res.message[:200]}"})})

    if t.want("tool_order", skip) and t.accepts.get("tools"):
        stage("tool_order")
        calls = [{"id": f"call_{c}", "type": "function", "function": {"name": "noop", "arguments": "{}"}} for c in "ab"]
        head = [{"role": "user", "content": "Call noop twice."}, {"role": "assistant", "content": "", "tool_calls": calls}]
        tool = lambda c: {"role": "tool", "tool_call_id": f"call_{c}", "content": "done"}  # noqa: E731
        media = {"role": "user", "content": [{"type": "text", "text": "Tool output image."}, img(data_uri(fx.tiny))]}
        ask = {"role": "user", "content": "Reply with OK."}
        contiguous = await call(http, t, head + [tool("a"), tool("b"), media, ask], PARAMS["tools"])
        interleaved = await call(http, t, head + [tool("a"), media, tool("b"), ask], PARAMS["tools"])
        done("tool_order", {"contiguous_ok": known(contiguous), "interleaved_ok": known(interleaved),
                            **({} if interleaved.ok else {"message": interleaved.message[:300]})})

    if t.want("video_density", skip):
        stage("video_density")
        try:
            done("video_density", await density(http, t, fx, args.density_videos, args.density_repeats))
        except Exception as e:  # noqa: BLE001
            done("video_density", {"error": f"{type(e).__name__}: {e}"[:300]})

    if t.want("video_frames", skip):
        stage("video_frames")
        dens = t.out.get("video_density") or {}
        which = "short_360p" if any("360p" in v for v in dens.get("video_used", [])) else "short"
        try:
            done("video_frames", await frame_calibration(http, t, fx, which, dens))
        except Exception as e:  # noqa: BLE001
            done("video_frames", {"error": f"{type(e).__name__}: {e}"[:300]})

    if t.want("video_bytes", skip):
        stage("video_bytes")
        try:
            mp4 = fx.base_mp4()
            small = await fx.url("video_base.mp4", lambda: mp4, "video/mp4")
            vid = lambda u: [{"role": "user", "content": [{"type": "text", "text": "A video:"},  # noqa: E731
                                                          {"type": "video_url", "video_url": {"url": u}}]},
                             {"role": "user", "content": "Reply with OK."}]
            CTX.set({"dimension": "video_bytes", "value": "base"})
            first = await call(http, t, vid(small))
            res: dict[str, Any] = {"video_input": known(first), "base_bytes": len(mp4)}
            if not first.ok and first.kind in INLINE_RETRY:
                # URL refused: check the same clip inline, so "no video by URL" isn't
                # reported as "no video". The size search needs URLs, so it stops here.
                CTX.set({"dimension": "video_bytes", "value": "base_inline"})
                inline = await call(http, t, vid(f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}"))
                last = first if inline.ok else inline      # when both fail, the inline answer is the real one
                res.update({"video_input": known(inline), "video_url": False, "inline_only": inline.ok,
                            "url_kind": first.kind, "kind": last.kind, "message": last.message[:300],
                            "request_id": last.request_id})
                if inline.ok:
                    # The same size search, with the padded clip sent inline (base64). Base64 adds a third,
                    # so the inline request-body cap allows video up to 3/4 of it.
                    CTX.set({"dimension": "video_bytes", "value": "inline"})

                    async def probe_video_inline(v: float) -> Result:
                        uri = f"data:video/mp4;base64,{base64.b64encode(pad_mp4(mp4, int(v * MB))).decode()}"
                        return await call(http, t, vid(uri))
                    res["bytes"] = await search(probe_video_inline, 2, args.inline_cap_mb * 0.75, integer=False, tol=0.25)
                    res["sent_inline"] = True
                    res["inline_cap_mb"] = args.inline_cap_mb * 0.75
            elif first.ok:
                CTX.set({"dimension": "video_bytes"})

                async def probe_video(v: float) -> Result:
                    size = int(v * MB)
                    try:
                        u = await fx.url(f"video_{size}.mp4", lambda: pad_mp4(mp4, size), "video/mp4")
                    except Exception as e:  # noqa: BLE001 - our upload failed: inconclusive, keep partial results
                        return Result(False, 0, "transient", None, f"fixture upload failed: {type(e).__name__}: {e}"[:300])
                    return await call(http, t, vid(u))
                res["bytes"] = await search(probe_video, 2, 512, integer=False, tol=0.25)
            else:
                res.update({"kind": first.kind, "message": first.message[:300], "request_id": first.request_id})
            done("video_bytes", res)
        except Exception as e:  # noqa: BLE001
            done("video_bytes", {"error": f"{type(e).__name__}: {e}"[:300]})

    count_max = None
    if t.want("image_count", skip) and t.accepts.get("image_inline"):
        stage("image_count")
        tiny = data_uri(fx.tiny)
        res = await search(lambda n: call(http, t, image_messages([tiny] * int(n))), 16, args.image_cap, integer=True, tol=0)
        count_max = res.get("max_ok")
        done("image_count", res)

    n_pad = max(1, min(8, count_max or 8))
    if t.want("inline_bytes", skip) and t.accepts.get("image_inline"):
        stage("inline_bytes")

        async def probe_inline(v: float) -> Result:
            uri = data_uri(pad_jpeg(fx.tiny, int(v * MB / n_pad)))
            return await call(http, t, image_messages([uri] * n_pad))
        res = await search(probe_inline, 2, args.inline_cap_mb, integer=False, tol=0.2)
        res["images"] = n_pad
        done("inline_bytes", res)

    if t.want("url_bytes", skip) and t.accepts.get("image_url"):
        stage("url_bytes")

        async def probe_url(v: float) -> Result:
            each = int(round(v * MB / n_pad / 250_000)) * 250_000 or 250_000   # quantized so uploads are reused
            try:
                urls = [await fx.url(f"pad_{each}_{i}.jpg", lambda: pad_jpeg(fx.tiny, each), "image/jpeg")
                        for i in range(n_pad)]
            except Exception as e:  # noqa: BLE001
                return Result(False, 0, "transient", None, f"fixture upload failed: {type(e).__name__}: {e}"[:300])
            return await call(http, t, image_messages(urls))
        res = await search(probe_url, 2, 128, integer=False, tol=0.2)
        res["images"] = n_pad
        done("url_bytes", res)

    if t.want("frames_768", skip) and t.accepts.get("image_inline"):
        stage("frames_768")
        frame = data_uri(fx.frame)
        res = await search(lambda n: call(http, t, image_messages([frame] * int(n))), 32, 384, integer=True, tol=0.15)
        toks = [x for x in res["tried"] if x.get("ok") and x.get("prompt_tokens")]
        if toks:
            best = max(toks, key=lambda x: x["value"])
            res["tokens_per_frame"] = round(best["prompt_tokens"] / best["value"])
        res["frame"] = "768x432"
        done("frames_768", res)


# --------------------------------------------------------------------------
# OpenRouter: upstream expansion and credit guard
# --------------------------------------------------------------------------

async def openrouter_upstreams(http: aiohttp.ClientSession, base_url: str, key: str, model: str) -> list[dict]:
    async with http.get(f"{base_url.rstrip('/')}/models/{model}/endpoints", headers={"Authorization": f"Bearer {key}"}) as r:
        r.raise_for_status()
        return ((await r.json()).get("data") or {}).get("endpoints") or []


class CreditGuard:
    """Stops OpenRouter targets when the account balance drops below `floor` USD."""

    def __init__(self, http: aiohttp.ClientSession, key: Optional[str], floor: float) -> None:
        self.http, self.key, self.floor = http, key, floor
        self._at, self._left = 0.0, None
        self._lock = asyncio.Lock()

    async def remaining(self) -> Optional[float]:
        if not self.key:
            return None
        async with self._lock:
            if time.time() - self._at > 10:
                try:
                    async with self.http.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {self.key}"}) as r:
                        d = (await r.json()).get("data") or {}
                        self._left = float(d.get("total_credits", 0)) - float(d.get("total_usage", 0))
                except Exception:  # noqa: BLE001
                    pass
                self._at = time.time()
        return self._left

    async def ok(self) -> bool:
        left = await self.remaining()
        return left is None or left >= self.floor


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def fmt_limit(res: Optional[dict], unit: str = "") -> str:
    if not res:
        return "—"
    lo, hi = res.get("max_ok"), res.get("fail_at")
    f = (lambda x: f"{x:g}{unit}") if unit else (lambda x: f"{x}")
    if res.get("inconclusive"):
        return f"≥{f(lo)} (then {res['inconclusive']})" if lo else f"? ({res['inconclusive']})"
    if hi is None:
        return f"≥{f(lo)}" if lo is not None else "—"
    stated = f", states {res['stated_limit']}" if res.get("stated_limit") else ""
    return f"{f(lo)}–{f(hi)} ({res.get('fail_kind')}{stated})" if lo else f"<{f(hi)} ({res.get('fail_kind')}{stated})"


def fmt_density(d: Optional[dict]) -> str:
    if not d:
        return "—"
    if d.get("visual_tokens_per_min") is not None:
        s = f"{d['visual_tokens_per_min']:,.0f} tok/min"
        if d.get("est_frames_per_min") is not None:
            s += f" (≈{d['est_frames_per_min']:g} frames/min)"
        if d.get("spread_pct"):
            s += f" ±{d['spread_pct']:g}%"
        return s
    run = next((r for r in d.get("runs", []) if not r.get("ok")), None)
    return f"✗ {run.get('kind')}" if run else (d.get("error", "—")[:40])


def summary_row(t: Target) -> list[str]:
    o = t.out
    if not o.get("baseline", {}).get("ok"):
        b = o.get("baseline", {})
        return [t.name, f"❌ baseline: {b.get('kind') or o.get('skipped') or '—'}", "", "", "", "", "", "", "", ""]
    p = o.get("params", {})
    rejected = [k for k, v in p.items() if v.get("ok") is False]
    v = o.get("video_bytes", {})
    video = ("no video" if v and not v.get("video_input") else fmt_limit(v.get("bytes"), "MB")) if v else "—"
    to = o.get("tool_order")
    order = "—" if not to else ("strict" if to["contiguous_ok"] and to["interleaved_ok"] is False else
                                "lenient" if to["interleaved_ok"] else "?")
    f7 = o.get("frames_768") or {}
    frames = fmt_limit(f7) + (f" · {f7['tokens_per_frame']} tok/frame" if f7.get("tokens_per_frame") else "")
    ii = o.get("image_input", {})
    transport = "/".join(x for x, ok in (("inline", ii.get("inline")), ("url", ii.get("url"))) if ok) or "none"
    return [t.name, transport, fmt_limit(o.get("image_count")), fmt_limit(o.get("inline_bytes"), "MB"),
            fmt_limit(o.get("url_bytes"), "MB"), video, fmt_density(o.get("video_density")), frames, order,
            ", ".join(rejected) or "none"]


def write_outputs(targets: list[Target], out_dir: Path, started: float) -> Path:
    (out_dir / "probe.json").write_text(json.dumps(
        [{"name": t.name, "base_url": t.base_url, "model": t.model, "pin": t.pin, "service": t.service,
          "requests": t.requests, "prompt_tokens": t.prompt_tokens, "probe_version": PROBE_VERSION,
          "results": t.out} for t in targets], indent=1, default=str))

    head = ["Target", "Image transport", "Image count", "Inline bytes", "URL bytes", "Video file", "Video density",
            "768×432 frames", "Tool order", "Rejected fields"]
    lines = [f"# Provider limit probe · {datetime.now():%Y-%m-%d %H:%M}", "",
             f"{len(targets)} targets · {sum(t.requests for t in targets)} requests · "
             f"{sum(t.prompt_tokens for t in targets):,} prompt tokens (where reported) · "
             f"{(time.time() - started) / 60:.0f} min · probe v{PROBE_VERSION}", "",
             "Ranges are `largest pass – first failure (failure kind, limit stated in the error)`. "
             "`≥` means no failure up to the probe cap. Video density: visual tokens billed per minute of the "
             "reference video (exact), and ≈ frames/min from a per-model tokens-per-frame constant (estimate).", "",
             "Every request and response is in `requests.jsonl` (look up a `request_id`).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(c.replace("|", "/") for c in summary_row(t)) + " |" for t in targets]
    lines += ["", "## Errors (evidence)", ""]
    for t in targets:
        for dim, res in t.out.items():
            if isinstance(res, dict):
                for c in [res] + [v for v in res.values() if isinstance(v, dict)]:
                    if c.get("message") and (c.get("fail_kind") or c.get("kind") or c.get("inconclusive")):
                        rid = f" · request `{c['request_id']}`" if c.get("request_id") else ""
                        lines.append(f"- **{t.name} · {dim}** ({c.get('fail_kind') or c.get('kind') or c.get('inconclusive')}){rid}: "
                                     f"`{c['message'][:240].replace('`', '')}`")
                        break
    path = out_dir / "probe.md"
    path.write_text("\n".join(lines) + "\n")
    return path


YES_NO_DIMS = {"image_input", "tool_order"}
SEARCH_DIMS = {"image_count", "inline_bytes", "url_bytes", "frames_768"}
# Kinds that came from matching the error text, so they can be re-derived from the stored
# message when the patterns improve. transient/timeout come from the HTTP status: kept as is.
RELABEL = {"other", "invalid_media", "no_route", "no_video", "request_bytes"}


def relabel(t: Target, results: dict) -> dict:
    """Re-classify stored failures with the current PATTERNS, so every run is judged by
    the same rules (e.g. Baseten's "Too many images: 97 (max 96)" was once "other")."""
    def kind_of(kind: Optional[str], msg: Optional[str]) -> tuple[Optional[str], Optional[int]]:
        if kind not in RELABEL or not msg:
            return kind, None
        new, stated = classify_for(t, 400, msg)
        # Only ever more specific: stored messages are truncated, so re-reading one can miss
        # what the full message showed (e.g. the pinned upstream in OpenRouter's filter list).
        if new == "other" or (new == "no_route" and kind != "no_route"):
            return kind, None
        return new, stated

    out = json.loads(json.dumps(results))
    for dim in SEARCH_DIMS | {"video_bytes"}:
        res = out.get(dim)
        if dim == "video_bytes" and isinstance(res, dict):
            k, _ = kind_of(res.get("kind"), res.get("message"))
            if k != res.get("kind"):
                res["kind"] = k
                if res.get("video_input") is False and k in INCONCLUSIVE:
                    res["video_input"] = None
            res = res.get("bytes")
            holder = out[dim]
        else:
            holder = out
        if not isinstance(res, dict) or not res.get("fail_kind"):
            continue
        k, stated = kind_of(res["fail_kind"], res.get("message"))
        above = [x for x in res.get("tried", []) if not x.get("ok") and x["value"] > (res.get("max_ok") or 0)]
        if k in INCONCLUSIVE and k != "no_route" and len({x["value"] for x in above}) < 2:
            # A one-off server failure set the upper bound: keep only the lower bound. (Failures
            # at two or more sizes above every pass are a reproducible ceiling, kept as measured.)
            new = {**{x: v for x, v in res.items() if x not in ("fail_at", "fail_kind", "stated_limit")},
                   "fail_at": None, "inconclusive": k, "relabelled_from": res["fail_kind"]}
        else:
            new = {**res, "fail_kind": k, "stated_limit": res.get("stated_limit") or stated}
        if dim == "video_bytes":
            holder["bytes"] = new
        else:
            out[dim] = new
    for dim in ("video_density",):
        for run in (out.get(dim) or {}).get("runs", []):
            run["kind"], _ = kind_of(run.get("kind"), run.get("message"))
    vf = out.get("video_frames")
    if isinstance(vf, dict) and vf.get("kind"):
        vf["kind"], _ = kind_of(vf["kind"], vf.get("message"))
    for name, v in (out.get("params") or {}).items():
        k, _ = kind_of(v.get("kind"), v.get("message"))
        if k != v.get("kind"):
            v["kind"] = k
            if v.get("ok") is False and k in INCONCLUSIVE:
                v["ok"] = None
    ii = out.get("image_input") or {}
    for side in ("inline", "url"):
        err = ii.get(f"{side}_error")
        if ii.get(side) is False and err and ": " in err:
            k0, msg = err.split(": ", 1)
            k, _ = kind_of(k0, msg)
            if k in INCONCLUSIVE:
                ii[side] = None
    return out


def merge_runs() -> Path:
    """Combine all probe runs (oldest first). Per target and dimension, a later
    result replaces an earlier one unless it is only an error or inconclusive and
    the earlier one was conclusive or got further."""
    def weak(r: Any) -> bool:
        return isinstance(r, dict) and (bool(r.get("error")) or bool(r.get("inconclusive"))
                                        # video_bytes keeps its search one level down
                                        or (isinstance(r.get("bytes"), dict) and bool(r["bytes"].get("inconclusive")))
                                        # a baseline lost to billing/rate limits says nothing about the provider
                                        or (r.get("ok") is False and r.get("kind") in INCONCLUSIVE))

    def passed(r: Any) -> float:
        x = r.get("bytes", r) if isinstance(r, dict) else None
        return (x.get("max_ok") or 0) if isinstance(x, dict) else 0

    merged: dict[str, Target] = {}
    runs = sorted(RESULTS.glob("probe-2*/probe.json"), key=lambda f: f.stat().st_mtime)
    for f in runs:
        for t in json.loads(f.read_text()):
            tgt = merged.setdefault(t["name"], Target(t["name"], t["base_url"], t["model"], "", t.get("pin"),
                                                      service=t.get("service") or service_of(t["base_url"])))
            tgt.requests += t.get("requests", 0)
            tgt.prompt_tokens += t.get("prompt_tokens", 0)
            for dim, res in relabel(tgt, t["results"]).items():
                # A search that stopped early doesn't replace a conclusive result, or one that got further.
                if dim in tgt.out and weak(res) and (not weak(tgt.out[dim]) or passed(res) < passed(tgt.out[dim])):
                    continue
                prev = tgt.out.get(dim)
                if isinstance(prev, dict) and isinstance(res, dict):
                    # A later rate-limited check ("unknown") doesn't erase an earlier real answer.
                    if dim in YES_NO_DIMS:
                        res = {**res, **{k: v for k, v in prev.items() if res.get(k) is None and v is not None}}
                    elif dim == "params":
                        res = {**res, **{k: v for k, v in prev.items()
                                         if (res.get(k) or {}).get("ok") is None and v.get("ok") is not None}}
                tgt.out[dim] = res
    out_dir = RESULTS / "probe-merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = min((f.stat().st_mtime for f in runs), default=time.time())
    return write_outputs(list(merged.values()), out_dir, started)


# --------------------------------------------------------------------------

async def load_targets(http: aiohttp.ClientSession, paths: list[Path], secrets: dict, only: list[str],
                       exclude: list[str]) -> list[Target]:
    seen, out = set(), []
    for p in paths:
        for c in tomllib.loads(p.read_text())["target"]:
            key = secrets.get(c["api_key_env"]) or os.environ.get(c["api_key_env"])
            if not key:
                console.print(f"[yellow]skip {c['name']}: no {c['api_key_env']}[/]")
                continue
            dims = set(c["dims"]) if c.get("dims") else None
            svc = service_of(c["base_url"])
            variants: list[tuple[str, Optional[str]]] = []
            if c.get("upstreams") == "all" and svc == "openrouter":
                for e in await openrouter_upstreams(http, c["base_url"], key, c["model"]):
                    variants.append((f"{c['name']} · {e.get('tag')}", e.get("tag")))
            else:
                pin = c.get("openrouter_provider")
                variants.append((c["name"], None if pin == "auto" else pin))
            for name, pin in variants:
                sig = (c["base_url"].rstrip("/"), c["model"], pin)
                low = name.lower()
                if sig in seen or (only and not any(o.lower() in low for o in only)) or any(x.lower() in low for x in exclude):
                    continue
                seen.add(sig)
                out.append(Target(name, c["base_url"], c["model"], key, pin, dims=dims, service=svc, rpm=c.get("rpm"),
                                  video_inline=bool(c.get("video_inline"))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", action="append", help="TOML target file(s); default targets.toml + targets_direct.toml + targets_private.toml (if present)")
    ap.add_argument("--only", action="append", default=[], help="target name contains (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], help="skip targets whose name contains this (repeatable)")
    ap.add_argument("--skip", action="append", default=[], choices=DIMENSIONS, help="dimension to skip (repeatable)")
    ap.add_argument("--concurrency", type=int, default=12, help="targets probed at once")
    ap.add_argument("--per-provider", type=int, default=3, help="targets at once per provider (per upstream on OpenRouter)")
    ap.add_argument("--density-videos", default="short", help="comma list of reference videos: short,long")
    ap.add_argument("--density-repeats", type=int, default=2)
    ap.add_argument("--density-timeout", type=int, default=300, help="seconds to wait for a video request")
    ap.add_argument("--density-start", choices=["small", "h264", "trim", "minute"],
                    help="measure density on this variant of the reference clip from the start (e.g. minute: "
                         "providers whose full clip exceeds the context window or the timeout)")
    ap.add_argument("--inline-cap-mb", type=float, default=32)
    ap.add_argument("--image-cap", type=int, default=1024,
                    help="image-count search cap; models that bill per image (Gemini ~258 tok) cost a lot near 1024")
    ap.add_argument("--openrouter-floor", type=float, default=2.0, help="stop OpenRouter targets below this balance (USD)")
    ap.add_argument("--merge", action="store_true", help="don't probe: combine results/probe-*/ into results/probe-merged/")
    args = ap.parse_args()
    args.density_videos = [v.strip() for v in args.density_videos.split(",") if v.strip()]
    global DENSITY_TIMEOUT, DENSITY_START
    DENSITY_TIMEOUT = args.density_timeout
    DENSITY_START = args.density_start

    if args.merge:
        console.print(f"report → {merge_runs()}")
        return

    load_dotenv(HERE / ".env", override=False)
    secrets = {k: v for k, v in dotenv_values(HERE / ".env.local").items() if v}
    defaults = [HERE / "targets.toml", HERE / "targets_direct.toml", HERE / "targets_private.toml"]
    files = [Path(p) for p in args.targets] if args.targets else [p for p in defaults if p.exists()]

    out_dir = RESULTS / f"probe-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    global LOG
    LOG = RequestLog(out_dir / "requests.jsonl")
    started = time.time()
    targets: list[Target] = []

    def log(t: Target, dim: str, res: Any) -> None:
        if dim == "baseline":
            msg = "ok" if res["ok"] else f"[red]failed: {res['kind']} · {res['message'][:160]}[/]"
        elif dim == "params":
            bad = [k for k, v in res.items() if v["ok"] is False]
            unk = [k for k, v in res.items() if v["ok"] is None]
            msg = (f"rejects {', '.join(bad)}" if bad else "accepts all fields") + (f" (unknown: {', '.join(unk)})" if unk else "")
        elif dim == "image_input":
            yn = lambda v: "ok" if v else "?" if v is None else "no"
            msg = f"inline={yn(res['inline'])} url={yn(res['url'])}"
        elif dim == "tool_order":
            yn = lambda v: "ok" if v else "?" if v is None else "no"
            msg = f"contiguous={yn(res['contiguous_ok'])} interleaved={yn(res['interleaved_ok']).replace('no', 'rejected')}"
        elif dim == "video_bytes":
            msg = ("no video input: " + str(res.get("kind"))) if not res.get("video_input") else fmt_limit(res.get("bytes"), "MB")
        elif dim == "video_density":
            msg = fmt_density(res)
        elif dim == "video_frames":
            msg = (f"read {res['frames_read']} of {res['clip_frames']} frames · {res['tokens_per_frame']} tok/frame"
                   + (f" · ≈{res['frames_per_min_calibrated']} frames/min on the reference clip" if res.get("frames_per_min_calibrated") else "")
                   if res.get("frames_read") else f"✗ {res.get('kind') or res.get('error', '')[:60]}")
        else:
            msg = fmt_limit(res, "MB" if dim.endswith("bytes") else "")
        console.print(f"[dim]{datetime.now():%H:%M:%S}[/] {t.name:52} {dim:13} {msg}", highlight=False)
        write_outputs(targets, out_dir, started)

    async def run() -> None:
        fx = Fixtures()
        conn = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=conn) as http:
            targets.extend(await load_targets(http, files, secrets, args.only, args.exclude))
            if not targets:
                sys.exit("no targets selected")
            console.print(f"[bold]Limit probe v{PROBE_VERSION}[/] · {len(targets)} targets · {args.concurrency} at once, "
                          f"{args.per_provider} per provider\n[dim]results → {out_dir}[/]")
            global GUARD
            guard = GUARD = CreditGuard(http, secrets.get("OPENROUTER_API_KEY"), args.openrouter_floor)
            left = await guard.remaining()
            if left is not None:
                console.print(f"[dim]OpenRouter balance ${left:.2f}; OpenRouter targets stop below ${args.openrouter_floor:.2f}[/]")
            sem = asyncio.Semaphore(args.concurrency)
            per: dict[str, asyncio.Semaphore] = {}

            async def one(t: Target) -> None:
                lane = f"{t.service}:{t.pin}" if t.service == "openrouter" else t.service
                async with sem, per.setdefault(lane, asyncio.Semaphore(args.per_provider)):
                    if t.service == "openrouter" and not await guard.ok():
                        t.out["baseline"] = {"ok": False, "kind": "credit_guard", "message": "OpenRouter balance below floor"}
                        console.print(f"[yellow]skip {t.name}: OpenRouter balance below ${args.openrouter_floor:.2f}[/]")
                        return
                    try:
                        await probe_target(http, t, fx, set(args.skip), log, args)
                    except Exception as e:  # noqa: BLE001 - one target's crash shouldn't stop the rest
                        t.out["crash"] = f"{type(e).__name__}: {e}"[:300]
                        console.print(f"[red]{t.name}: crashed: {t.out['crash']}[/]")
                    console.print(f"[green]done[/] {t.name} · {t.requests} requests")
            await asyncio.gather(*(one(t) for t in targets))
            left = await guard.remaining()
            if left is not None:
                console.print(f"[dim]OpenRouter balance now ${left:.2f}[/]")

    try:
        asyncio.run(run())
    finally:
        report = write_outputs(targets, out_dir, started)
        console.print(f"\nreport → {report}\nrequests → {out_dir / 'requests.jsonl'}")


if __name__ == "__main__":
    main()
