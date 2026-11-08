import dataset from "../../data/capabilities.json";
import changesData from "../../data/changes.json";

export type Status = "measured" | "at_least" | "stated" | "estimated" | "inconclusive" | "unsupported" | "rejected" | "ignored" | "not_tested"
  | "varies" | "not_determined";

export interface Limit {
  status: Status;
  value?: number;
  fails_at?: number;
  stated?: number;
  at_least?: number;
  failure?: string;
  reason?: string;
  evidence?: string | null;
  source?: string;
  note?: string;
  video?: string;
  repeats?: number;
  spread_pct?: number | null;
  tokens_per_frame?: number;
  frames_per_video?: number;
  video_minutes?: number;
  frames_per_min_short_clip?: number;
  capped?: boolean;
  tokens_per_frame_assumed?: number;
  per_image?: number;
  // video_frames_per_min (billing-only frame curve)
  method?: string;
  unit?: string;
  units_per_video?: number;
  tokens_per_unit?: number;
  frame_size?: string;
  clip?: string;
  reference?: boolean;
  setting?: string;
  check?: { clip?: string; predicted_tokens_per_min: number; measured_tokens_per_min: number; diff_pct: number };
  // video_tokens_per_min on a route that refused the reference clip: the longest clip it took
  on_clip?: { seconds: number; value: number };
}

export interface Entry {
  id: string;
  provider: { id: string; name: string; via: string | null; tag?: string };
  model: { id: string; name: string; vendor: string | null; api_id: string; input_modalities?: string[] | null };
  measured_at: string;
  limits: Record<string, Limit>;
  accepts: { image_inline?: boolean; image_url?: boolean; video?: "yes" | "no" | "ignored" | null; request_fields: Record<string, boolean> };
  behaviour: { strict_tool_message_order: boolean | null };
  notes: string[];
  featured: boolean;
}

export interface Change {
  date: string;
  id: string;
  change: "added" | "removed" | "limit";
  limit?: string;
  from?: Partial<Limit>;
  to?: Partial<Limit>;
}

export const data = dataset as unknown as { updated: string; limits: Record<string, { label: string; unit: string }>; entries: Entry[] };
export const entries: Entry[] = data.entries;
export const changes: Change[] = (changesData as Change[]).slice().sort((a, b) => b.date.localeCompare(a.date));
export const updated: string = data.updated;

/** Columns shown in the main grid, in order. */
export const COLUMNS: { key: string; short: string; help: string }[] = [
  { key: "images_per_request", short: "Images", help: "Images per request, using tiny 64×64 images so only the count matters." },
  { key: "frames_768x432", short: "768×432 frames", help: "Real-size video frames per request. Whatever limit hits first: count, pixels or the context window." },
  { key: "tokens_per_frame_768x432", short: "Tokens / frame", help: "Prompt tokens per 768×432 frame." },
  { key: "inline_request_bytes", short: "Inline body", help: "Request body size with base64 images (padding only, almost no pixels). Probe cap: 16 MB." },
  { key: "url_image_bytes", short: "Images by URL", help: "Total size of images the provider downloads from URLs. Probe cap: 128 MB." },
  { key: "video_file_bytes", short: "Video file", help: "Size of a video sent as video_url. Probe cap: 512 MB." },
  { key: "video_tokens_per_min", short: "Video tok/min", help: "Visual tokens billed per minute of a real 5-minute reference clip: how much of the video the model is actually fed." },
  { key: "video_frames_per_min", short: "Frames/min", help: "Frames the model receives per minute of a 1080p video, worked out from the provider's billing and checked against the model's own preprocessing code. No model output is used." },
];

const UNIT: Record<string, "count" | "bytes" | "tokens"> = {
  images_per_request: "count",
  frames_768x432: "count",
  tokens_per_frame_768x432: "tokens",
  inline_request_bytes: "bytes",
  url_image_bytes: "bytes",
  video_file_bytes: "bytes",
  video_tokens_per_min: "tokens",
  video_frames_per_min: "count",
};

/** Below these values a limit is "tight" for video work. */
const TIGHT: Record<string, number> = {
  images_per_request: 200,
  frames_768x432: 200,
  inline_request_bytes: 10_000_000,
  url_image_bytes: 20_000_000,
  video_file_bytes: 25_000_000,
};
const SEVERE: Record<string, number> = { images_per_request: 16, frames_768x432: 16 };

export function fmtNumber(key: string, v: number): string {
  if (UNIT[key] === "bytes") {
    const mb = v / 1_000_000;
    return mb >= 1000 ? `${(mb / 1000).toLocaleString("en-US", { maximumFractionDigits: 2 })} GB`
      : `${mb.toLocaleString("en-US", { maximumFractionDigits: mb < 10 ? 1 : 0 })} MB`;
  }
  return v.toLocaleString("en-US");
}

export interface Cell {
  text: string;
  cls: string;
  title: string;
  sort: number;
}

/** How one limit renders in a table cell. */
export function cell(key: string, l: Limit | undefined, e?: Entry): Cell {
  if (!l || l.status === "not_tested") return { text: "—", cls: "na", title: "Not tested", sort: -1 };
  const v = l.value ?? l.at_least;
  const tight = v !== undefined && TIGHT[key] !== undefined && v < TIGHT[key];
  const severe = v !== undefined && SEVERE[key] !== undefined && v < SEVERE[key];
  const tone = severe ? " severe" : tight ? " tight" : "";
  const ev = l.evidence ? `\n\nProvider said: “${l.evidence}”` : "";
  if (key === "video_tokens_per_min" && l.on_clip && l.status !== "measured") {
    // Shown, but scored as the status says (rejected: 0; inconclusive: skipped): per-minute rates from
    // clips of different lengths don't compare when a provider sends a fixed number of frames per video.
    return { text: `${fmtNumber(key, l.on_clip.value)} †`, cls: l.status === "rejected" ? "unsupported" : "inconclusive", sort: l.on_clip.value,
      title: `Refused the 5-minute reference clip; ${fmtNumber(key, l.on_clip.value)} video tokens per minute on a `
        + `${l.on_clip.seconds}-second clip, the longest it accepted.${ev}` };
  }
  switch (l.status) {
    case "measured": {
      if (key === "video_frames_per_min") {
        return { text: `${fmtNumber(key, l.value ?? 0)}${l.capped ? " ⁎" : ""}`, cls: "measured", sort: l.value ?? 0,
          title: frameLines(l).join("\n") };
      }
      if (!l.value && l.fails_at !== undefined) {
        // Failed at the smallest size probed: all we know is the upper bound.
        return { text: `< ${fmtNumber(key, l.fails_at)}`, cls: `measured${tight || SEVERE[key] !== undefined ? " severe" : " tight"}`,
          title: `Failed even at ${fmtNumber(key, l.fails_at)}, the smallest size we tried.${ev}`, sort: 0.1 };
      }
      const range = l.fails_at !== undefined ? `Passes at ${fmtNumber(key, l.value ?? 0)}, fails at ${fmtNumber(key, l.fails_at)}` : "";
      const stated = l.stated ? `\nThe error states a limit of ${l.stated.toLocaleString("en-US")}.` : "";
      const extra = [l.note, l.video ? `Clip: ${l.video}${l.repeats ? `, ${l.repeats} runs` : ""}${l.spread_pct ? `, ±${l.spread_pct}%` : ""}` : ""].filter(Boolean).join("\n");
      const each = l.per_image ? ` (${fmtNumber(key, l.per_image)}/image)` : "";
      return { text: fmtNumber(key, l.value ?? 0) + each, cls: `measured${tone}`, title: [range + stated + ev, extra].filter(Boolean).join("\n").trim(), sort: l.value ?? 0 };
    }
    case "estimated":
      return { text: `≈ ${fmtNumber(key, l.value ?? 0)}`, cls: "measured", sort: l.value ?? 0,
        title: `Estimate: tokens per minute ÷ ${l.tokens_per_frame_assumed ?? "?"} tokens per video frame for this model family.` };
    case "ignored":
      return { text: "Ignored", cls: "measured tight", sort: -0.4,
        title: "Accepted with HTTP 200, but the video added no tokens: the model never saw it." };
    case "rejected":
      return { text: "Rejected", cls: "unsupported", sort: -0.6, title: `${l.note ?? "The reference video was rejected as too large."}${ev}` };
    case "varies":
      return { text: "Varies", cls: "inconclusive", sort: -0.3,
        title: e?.provider.id === "auto" ? (l.note ?? "Depends on the upstream OpenRouter picks.")
          : `The same request billed different amounts of video on different calls, so there is no single frame count.\n${l.note ?? ""}`.trim() };
    case "not_determined":
      return { text: "Not determined", cls: "inconclusive", sort: -0.3,
        title: `${l.note ?? "Billing does not match any preprocessing setting exactly."}${e?.limits.video_tokens_per_min?.value ? `\nVideo tokens per minute: ${fmtNumber("video_tokens_per_min", e.limits.video_tokens_per_min.value)}.` : ""}` };
    case "stated":
      return { text: fmtNumber(key, l.value ?? 0), cls: `measured${tone}`, title: `Limit stated by the provider${l.source ? ` (seen in ${l.source})` : ""}.${ev}`, sort: l.value ?? 0 };
    case "at_least":
      if (key === "video_frames_per_min")
        return { text: `≥ ${fmtNumber(key, l.value ?? 0)}`, cls: "atleast", sort: (l.value ?? 0) + 0.5, title: frameLines(l).join("\n") };
      return { text: `≥ ${fmtNumber(key, l.value ?? 0)}`, cls: "atleast", title: "No failure up to the probe's cap; the real limit is higher.", sort: (l.value ?? 0) + 0.5 };
    case "inconclusive":
      return {
        text: l.at_least ? `≥ ${fmtNumber(key, l.at_least)}?` : "?",
        cls: "inconclusive",
        title: `The search stopped on a ${l.reason ?? "timeout"}${l.at_least ? ` after ${fmtNumber(key, l.at_least)} passed` : ""}.`,
        sort: l.at_least ?? 0,
      };
    case "unsupported":
      return { text: key === "video_file_bytes" ? "No video" : "No", cls: "unsupported", title: `Not accepted.${ev}`, sort: -0.5 };
  }
  return { text: "—", cls: "na", title: "", sort: -1 };
}

/** Tooltip lines for a billing-derived frame count. */
function frameLines(l: Limit): string[] {
  const group = l.unit && l.unit !== "frame" ? ` in ${l.units_per_video} ${l.unit}s` : "";
  return [
    `${l.status === "at_least" ? "At least " : ""}${l.frames_per_video} frames from ${l.clip ?? `a ${l.video_minutes}-minute video`}${group}, `
      + `${l.tokens_per_unit} tokens each${l.frame_size ? ` (frames resized to ${l.frame_size.replace("x", "×")})` : ""}.`,
    l.reference ? "Same as the model's reference preprocessing." : l.setting ? `Provider setting: ${l.setting}.` : "",
    l.capped ? "⁎ A fixed number of frames per video: longer videos get fewer frames per minute." : "",
    l.check ? `Check: this predicts ${fmtNumber("video_tokens_per_min", l.check.predicted_tokens_per_min)} video tokens per minute on the reference clip; `
      + `the provider billed ${fmtNumber("video_tokens_per_min", l.check.measured_tokens_per_min)}.` : "",
    l.note ?? "",
  ].filter(Boolean);
}

export function providerLabel(e: Entry): string {
  return e.provider.via === "openrouter" ? `${e.provider.name} via OpenRouter` : e.provider.name;
}

export function providerSlug(e: Entry): string {
  return e.provider.via ? `${e.provider.via}-${e.provider.id}` : e.provider.id;
}

export function modelSlug(e: Entry): string {
  return e.model.id.split("/").pop()!.toLowerCase().replace(/[^a-z0-9.]+/g, "-");
}

export function rejectedFields(e: Entry): string[] {
  return Object.entries(e.accepts.request_fields || {}).filter(([, ok]) => ok === false).map(([k]) => k);
}

export function groupBy<T>(xs: T[], key: (x: T) => string): Map<string, T[]> {
  const m = new Map<string, T[]>();
  for (const x of xs) {
    const k = key(x);
    if (!m.has(k)) m.set(k, []);
    m.get(k)!.push(x);
  }
  return m;
}

export const LIMIT_LABELS: Record<string, string> = Object.fromEntries(
  Object.entries(data.limits).map(([k, v]) => [k, v.label]),
);

export function describeChange(c: Change): string {
  const e = entries.find((x) => x.id === c.id);
  const who = e ? `${e.model.name} on ${providerLabel(e)}` : c.id;
  if (c.change === "added") return `${who} added`;
  if (c.change === "removed") return `${who} removed`;
  const f = (l?: Partial<Limit>) =>
    !l || !l.status ? "nothing" : l.status === "unsupported" ? "not supported"
      : `${l.status === "at_least" ? "≥ " : ""}${l.value !== undefined ? fmtNumber(c.limit!, l.value) : l.status}`;
  return `${who}: ${LIMIT_LABELS[c.limit!] ?? c.limit} changed from ${f(c.from)} to ${f(c.to)}`;
}

// --------------------------------------------------------------------------
// Charts
// --------------------------------------------------------------------------

export interface Metric {
  key: string;
  tab: string;
  question: string;
  explain: string;
  /** A labelled reference line, in the metric's unit. */
  ref?: { value: number; label: string };
}

export const METRICS: Metric[] = [
  {
    key: "video_tokens_per_min", tab: "Video density",
    question: "How much of a video does the model actually see?",
    explain: "Every route got the same 5-minute 1080p video. Video tokens per minute is what the provider billed for it: the amount of visual input the model got, and what ranks providers. Frames per minute is how many frames that was, and how large, worked out from the bills and the model's own preprocessing code.",
  },
  {
    key: "frames_768x432", tab: "Frames per request",
    question: "How many video frames fit in one request?",
    explain: "768×432 frames, the size most video pipelines send. Whichever limit hits first counts: image count, decoded pixels or the context window.",
    ref: { value: 300, label: "100 overview frames + 2 clips" },
  },
  {
    key: "images_per_request", tab: "Images per request",
    question: "How many images can one request carry?",
    explain: "Tiny 64×64 images, so only the count limit is tested. Real frames can hit other limits sooner; see Frames per request.",
  },
  {
    key: "video_file_bytes", tab: "Video file size",
    question: "How large a video file can you send?",
    explain: "A video sent as a video_url. Most of these models only take images; those show No video.",
    ref: { value: 50_000_000, label: "50 MB" },
  },
  {
    key: "inline_request_bytes", tab: "Inline request body",
    question: "How big can the request body be with inline images?",
    explain: "Base64 images inside the request. Capped at 16 MB by our probe; ≥ means it accepted everything we sent.",
  },
  {
    key: "url_image_bytes", tab: "Images by URL",
    question: "How much image data will the provider download from URLs?",
    explain: "Images sent as links the provider fetches itself. Capped at 128 MB by our probe.",
  },
  {
    key: "tokens_per_frame_768x432", tab: "Tokens per frame",
    question: "How many tokens does one frame cost?",
    explain: "Prompt tokens per 768×432 frame. Lower is cheaper: a 100-frame clip costs 100× this.",
  },
];

export interface Bar {
  entry: Entry;
  value: number | null;   // null: no bar (unsupported / not tested / unknown)
  open: boolean;           // the real limit is higher than `value`
  text: string;
  cls: string;
  tip: string;
}

export function bars(key: string, list: Entry[] = entries): Bar[] {
  const out = key === OVERALL ? list.map((e): Bar => {
    const o = overall(e);
    return o
      ? { entry: e, value: o.score, open: false, text: String(o.score), cls: "measured",
          tip: `On average ${o.score}% of the best ${e.model.name} provider's limit, over the ${o.limits} limits measured.` }
      : { entry: e, value: null, open: false, text: "—", cls: "na", tip: "Too few limits measured to compare." };
  }) : list.map((e) => {
    const l = e.limits[key];
    const c = key === "video_frames_per_min" && (!l || l.status === "not_tested") ? videoVerdict(e) : cell(key, l, e);
    const v = l ? (l.status === "inconclusive" ? l.at_least ?? null
      : ["unsupported", "not_tested", "ignored", "rejected"].includes(l.status) ? null : l.value || null) : null;
    const open = !!l && (l.status === "at_least" || (l.status === "inconclusive" && l.at_least !== undefined));
    const tok = key === "video_frames_per_min" && e.limits.video_tokens_per_min?.value;
    const text = key === "video_frames_per_min" && v !== null ? `${c.text}/min${tok ? ` · ${fmtNumber("video_tokens_per_min", tok)} tok` : ""}` : c.text;
    return { entry: e, value: v, open, text, cls: c.cls, tip: c.title };
  });
  return out.sort((a, b) => barRank(b) - barRank(a) || a.entry.model.name.localeCompare(b.entry.model.name));
}

/** Frames per minute needs video to work and a billing frame curve. Without a value, say
 *  which: a pair whose video was refused or ignored was tested, not "not tested". */
function videoVerdict(e: Entry): Cell {
  const tok = e.limits.video_tokens_per_min;
  const ev = tok?.evidence ? `\n\nProvider said: “${tok.evidence}”` : "";
  switch (tok?.status) {
    case "unsupported":
      return { text: "No video", cls: "unsupported", sort: -0.5, title: `Video is not accepted on this route.${ev}` };
    case "ignored":
      return cell("video_frames_per_min", tok, e);
    case "measured":
      return { text: "Not measured", cls: "inconclusive", sort: -0.3,
        title: `Video works (${fmtNumber("video_tokens_per_min", tok.value ?? 0)} tokens per minute), but frames per minute `
          + "hasn't been measured on this route with the billing method yet." };
    case "rejected":
      return cell("video_tokens_per_min", tok, e);
    case "inconclusive":
      return { text: "?", cls: "inconclusive", sort: -0.3,
        title: `The video probe got no verdict (${tok.reason ?? tok.failure ?? "error"}).${ev}` };
  }
  return cell("video_frames_per_min", undefined, e);
}

// --------------------------------------------------------------------------
// Video density: tokens per minute (ranked) with frames delivered beside it
// --------------------------------------------------------------------------

export const DENSITY = "video_tokens_per_min";

/** A clip shorter than the 5-minute reference, in words. */
export function clipLabel(minutes: number): string {
  return Math.abs(minutes - 1) < 0.02 ? "the first minute" : Math.abs(minutes - 4) < 0.02 ? "the first 4 minutes"
    : `a ${Math.round(minutes * 60)} s clip`;
}
const VARIANT_LABEL: Record<string, string> = { baker_1min_h264: "the first minute", baker_4min_h264: "the first 4 minutes",
  baker_5min_360p: "a 360p copy" };

export interface DensityRow {
  entry: Entry;
  tok: number | null;        // video tokens per minute (bar), or null
  tokText: string;           // value, or what happened instead
  tokClip: string;           // "the first minute" etc., when not the full reference clip
  fr: number | null;         // frames delivered per minute (bar), or null
  frText: string;
  frSize: string;            // "1932×1092"
  frClip: string;
  fixed: boolean;            // a fixed number of frames per video
  noVideo: boolean;          // refused video: folded into one line
  refused: boolean;          // refused the 5-minute video; numbers are from a shorter clip (scored 0)
  untested: boolean;
  tip: string;
}

export function densityRow(e: Entry): DensityRow {
  const t = e.limits.video_tokens_per_min, f = e.limits.video_frames_per_min;
  const tokC = cell(DENSITY, t, e);
  const tok = t?.status === "measured" ? t.value ?? null : t?.on_clip ? t.on_clip.value : null;
  const tokClip = !t ? "" : t.status !== "measured" ? (t.on_clip ? clipLabel(t.on_clip.seconds / 60) : "")
    : VARIANT_LABEL[(t.video ?? "").split(",")[0].trim()] ?? "";
  const frOk = !!f && ["measured", "at_least"].includes(f.status) && !!f.value;
  const frText = frOk ? `${f!.status === "at_least" ? "≥ " : ""}${fmtNumber("video_frames_per_min", f!.value!)}`
    : f?.status === "varies" ? "Varies between identical requests"
    : f?.status === "not_determined" ? "Not determined"
    : tok !== null ? "Not measured" : "";
  const frClip = !frOk ? "" : f!.clip?.startsWith("the first") ? f!.clip.replace(/ of the reference clip.*/, "")
    : f!.clip?.includes("360p") ? "a 360p copy"
    : f!.video_minutes && f!.video_minutes < 5 ? clipLabel(f!.video_minutes) : "";
  const tokText = tok !== null ? fmtNumber(DENSITY, tok)
    : t?.status === "ignored" ? "Ignored: billed 0 tokens" : t?.status === "rejected" ? "Rejected"
    : t?.status === "inconclusive" ? "No verdict" : tokC.text;
  const tip = [
    tok !== null ? `${fmtNumber(DENSITY, tok)} video tokens per minute, billed by the provider${tokClip ? `, on ${tokClip}` : ""}.` : tokC.title,
    frOk ? frameLines(f!).join("\n") : f && f.status !== "not_tested" ? cell("video_frames_per_min", f, e).title : "",
  ].filter(Boolean).join("\n\n");
  return {
    entry: e, tok, tokText, tokClip, fr: frOk ? f!.value! : null, frText,
    frSize: frOk && f!.frame_size ? f!.frame_size.replace("x", "×") : "", frClip, fixed: frOk && !!f!.capped,
    noVideo: t?.status === "unsupported", refused: !!t && t.status !== "measured" && !!t.on_clip, untested: !t || t.status === "not_tested", tip,
  };
}

/** Density rows in score order: routes that took the video (most tokens first), then routes that
 *  refused it but took a shorter clip (shown, scored 0), then what happened instead. */
export function densityRows(list: Entry[]): DensityRow[] {
  const tier = (r: DensityRow) => (r.tok !== null ? (r.refused ? 3 : 4) : r.tokText.startsWith("Ignored") ? 2 : r.noVideo ? 0 : 1);
  return list.map(densityRow).sort((a, b) => tier(b) - tier(a) || (b.tok ?? 0) - (a.tok ?? 0) || (b.fr ?? 0) - (a.fr ?? 0)
    || a.entry.model.name.localeCompare(b.entry.model.name));
}

/** Sort key for a bar: higher is better; an open "≥" value edges out the same measured one. */
export function barRank(b: Bar): number {
  return b.value === null ? (b.text === "Ignored" ? -0.5 : b.cls.includes("unsupported") ? -1 : -2) : b.value + (b.open ? 0.5 : 0);
}

const ORDINAL = new Intl.PluralRules("en-US", { type: "ordinal" });
const SUFFIX: Record<string, string> = { one: "st", two: "nd", few: "rd", other: "th" };
export const ordinal = (n: number) => `${n}${SUFFIX[ORDINAL.select(n)]}`;

// Tokens per image aren't better or worse when higher, so they don't count toward standing.
// Video density counts as video tokens per minute: it grows with both the number and the size
// of the frames. Frames per minute is shown next to it, as context, but not scored.
const UNRANKED = new Set(["tokens_per_frame_768x432", "video_frames_per_min"]);
/** Limits where higher is better, in column order. */
export const RANKED: string[] = COLUMNS.map((c) => c.key).filter((k) => !UNRANKED.has(k));

/** A model's providers on one limit, best first; pairs not tested on it are left out. */
export function peers(key: string, e: Entry): Bar[] {
  return bars(key, entries.filter((x) => modelSlug(x) === modelSlug(e))).filter((b) => b.cls !== "na");
}

/** This pair's limit as a share of the best provider's for the same model (0–1), or
 *  null when it wasn't measured, no other provider was, or no provider accepts it.
 *  A rejected or unsupported input scores 0; an inconclusive one counts at the largest
 *  size that passed, or is left out if none did. */
// A route that accepts a video and bills 0 tokens for it shows the model nothing: the file it
// accepted is no capability, so it scores 0 on video file size and doesn't set the best.
const ignoresVideo = (e: Entry) => e.limits.video_tokens_per_min?.status === "ignored" || e.accepts.video === "ignored";

export function shareOfBest(key: string, e: Entry): number | null {
  const ps = key === "video_file_bytes" ? peers(key, e).filter((b) => b.entry === e || !ignoresVideo(b.entry)) : peers(key, e);
  if (key === "video_file_bytes" && ignoresVideo(e) && ps.length >= 2) return 0;
  const me = ps.find((b) => b.entry === e);
  const best = ps[0]?.value ?? null;
  // No provider takes it at all (e.g. video files on every DeepSeek endpoint): nothing to compare.
  if (!me || ps.length < 2 || best === null) return null;
  if (me.value === null) return /unsupported|tight/.test(me.cls) || ["Rejected", "Ignored"].includes(me.text) ? 0 : null;
  return best ? Math.min(1, me.value / best) : null;
}

/** The overall score that ranks a model's providers: its limits as a share of the best
 *  provider's, averaged over the limits it was measured on, 0–100. Needs two limits. */
const overallCache = new Map<Entry, { score: number; limits: number } | null>();
/** Models with no overall score, and why: their providers weren't measured on the same limits,
 *  so an average over whatever each one has would compare unlike things. (docs/TODO-reruns.md) */
export const NO_OVERALL: Record<string, string> = {
  "qwen3.8-27b": "Most of its OpenRouter providers were measured only on images per request and video density, so their scores would rest on 2 limits against 6 for the others. Compare providers on each limit instead.",
};
export function overall(e: Entry): { score: number; limits: number } | null {
  if (NO_OVERALL[modelSlug(e)]) return null;
  if (!overallCache.has(e)) {
    const s = RANKED.map((k) => shareOfBest(k, e)).filter((x): x is number => x !== null);
    overallCache.set(e, s.length >= 2 ? { score: Math.round((100 * s.reduce((a, b) => a + b, 0)) / s.length), limits: s.length } : null);
  }
  return overallCache.get(e)!;
}

/** The model in the homepage frames-per-request block, and the one in the video density hero. */
export const HERO_MODEL = "deepseek-v4.1-flash-1017";
export const VIDEO_MODEL = "kimi-k3.1-pro";

export const OVERALL = "overall";
/** One-line explanation of the overall score, for tooltips; /methodology#overall-score has the rest. */
export const OVERALL_HELP = "Each limit as a percentage of the best provider's for this model, averaged over the limits this provider was measured on. 100 = matches the best provider on every limit.";
export const OVERALL_METRIC: Metric = {
  key: OVERALL, tab: "Overall",
  question: "Which provider gives this model the most room overall?",
  explain: "Each limit as a percentage of the best provider's for this model, averaged over the limits that provider was measured on. Video density counts as video tokens per minute; tokens per image are left out. 100 means it matches the best provider on every limit. Providers measured on fewer than two limits aren't scored.",
};

/** A round axis maximum and 4–6 ticks covering `max`. */
export function niceScale(max: number): { max: number; ticks: number[] } {
  if (max <= 0) return { max: 1, ticks: [0, 1] };
  const raw = max / 4;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw)!;
  const top = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let t = 0; t <= top + 1e-9; t += step) ticks.push(t);
  return { max: top, ticks };
}

export function tickLabel(key: string, v: number): string {
  if (UNIT[key] === "bytes") return v === 0 ? "0" : `${(v / 1_000_000).toLocaleString("en-US")} MB`;
  return v.toLocaleString("en-US");
}

/** Range of a limit across entries, for the headline tiles. */
export function span(key: string): { lo: number; hi: number; hiOpen: boolean } | null {
  const bs = bars(key).filter((b) => b.value !== null);
  if (!bs.length) return null;
  const vals = bs.map((b) => b.value!);
  const hi = Math.max(...vals);
  return { lo: Math.min(...vals), hi, hiOpen: bs.some((b) => b.value === hi && b.open) };
}

export function models(list: Entry[] = entries): { slug: string; name: string; vendor: string | null; entries: Entry[] }[] {
  const g = groupBy(list, modelSlug);
  return [...g.entries()].map(([slug, es]) => ({ slug, name: es[0].model.name, vendor: es[0].model.vendor, entries: es }))
    .sort((a, b) => b.entries.length - a.entries.length || a.name.localeCompare(b.name));
}

export function providers(): { slug: string; name: string; entries: Entry[] }[] {
  const g = groupBy(entries, providerSlug);
  return [...g.entries()].map(([slug, es]) => ({ slug, name: providerLabel(es[0]), entries: es }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

export const featured: Entry[] = entries.filter((e) => e.featured);

export function videoLabel(e: Entry): { text: string; cls: string; tip: string } {
  switch (e.accepts.video) {
    case "yes": return { text: "Yes", cls: "yes", tip: "video_url accepted and the video reaches the model." };
    case "ignored": return { text: "Ignored", cls: "warn", tip: "Accepted with HTTP 200, but the video adds no tokens: the model never sees it." };
    case "no": return { text: "No", cls: "no", tip: "video_url rejected or not routable." };
    default: return { text: "Unknown", cls: "unk", tip: "Not determined (timeout or provider error)." };
  }
}
