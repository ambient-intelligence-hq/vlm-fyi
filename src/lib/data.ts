import dataset from "../../data/capabilities.json";
import changesData from "../../data/changes.json";

export type Status = "measured" | "at_least" | "stated" | "estimated" | "inconclusive" | "unsupported" | "rejected" | "ignored" | "not_tested";

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
  { key: "video_frames_per_min", short: "Frames/min", help: "Distinct frames the model receives per minute of the 5-minute clip, measured with a frame-counter calibration clip." },
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
  switch (l.status) {
    case "measured": {
      if (key === "video_frames_per_min") {
        const lines = [
          `${l.frames_per_video} distinct frames over the ${l.video_minutes}-minute clip.`,
          `Measured: ${l.tokens_per_frame} tokens per frame (the model read frame numbers from a 30 s counter clip: ${l.frames_per_min_short_clip}/min there).`,
          l.capped ? "Capped: the provider limits frames per video, so longer videos get fewer frames per minute." : "",
        ];
        return { text: `${fmtNumber(key, l.value ?? 0)}${l.capped ? " ⁎" : ""}`, cls: "measured", sort: l.value ?? 0,
          title: lines.filter(Boolean).join("\n") };
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
      return { text: "Rejected", cls: "unsupported", sort: -0.6, title: `The reference video was rejected as too large.${ev}` };
    case "stated":
      return { text: fmtNumber(key, l.value ?? 0), cls: `measured${tone}`, title: `Limit stated by the provider${l.source ? ` (seen in ${l.source})` : ""}.${ev}`, sort: l.value ?? 0 };
    case "at_least":
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
    key: "video_frames_per_min", tab: "Video density",
    question: "How much of a video does the model actually see?",
    explain: "Distinct frames the model receives per minute of the same 5-minute clip, measured with a frame-counter clip. ⁎ means the provider caps frames per video, so longer videos get fewer per minute.",
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

/** Frames per minute needs video to work and a frame calibration. Without a value, say
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
      return { text: "Not calibrated", cls: "inconclusive", sort: -0.3,
        title: `Video works (${fmtNumber("video_tokens_per_min", tok.value ?? 0)} tokens per minute), but the frame-counter `
          + "calibration did not complete, so frames per minute is unknown." };
    case "inconclusive":
      return { text: "?", cls: "inconclusive", sort: -0.3,
        title: `The video probe got no verdict (${tok.reason ?? tok.failure ?? "error"}).${ev}` };
  }
  return cell("video_frames_per_min", undefined, e);
}

/** Sort key for a bar: higher is better; an open "≥" value edges out the same measured one. */
export function barRank(b: Bar): number {
  return b.value === null ? (b.text === "Ignored" ? -0.5 : b.cls.includes("unsupported") ? -1 : -2) : b.value + (b.open ? 0.5 : 0);
}

const ORDINAL = new Intl.PluralRules("en-US", { type: "ordinal" });
const SUFFIX: Record<string, string> = { one: "st", two: "nd", few: "rd", other: "th" };
export const ordinal = (n: number) => `${n}${SUFFIX[ORDINAL.select(n)]}`;

// Token counts aren't better or worse when higher, so they don't count toward standing.
const UNRANKED = new Set(["tokens_per_frame_768x432", "video_tokens_per_min"]);
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
export function shareOfBest(key: string, e: Entry): number | null {
  const ps = peers(key, e);
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
export function overall(e: Entry): { score: number; limits: number } | null {
  if (!overallCache.has(e)) {
    const s = RANKED.map((k) => shareOfBest(k, e)).filter((x): x is number => x !== null);
    overallCache.set(e, s.length >= 2 ? { score: Math.round((100 * s.reduce((a, b) => a + b, 0)) / s.length), limits: s.length } : null);
  }
  return overallCache.get(e)!;
}

/** The model in the homepage hero (frames per request), and the one in the video density block below it. */
export const HERO_MODEL = "deepseek-v4.1-flash-1017";
export const VIDEO_MODEL = "qwen3.8-27b";

export const OVERALL = "overall";
/** One-line explanation of the overall score, for tooltips; /methodology#overall-score has the rest. */
export const OVERALL_HELP = "Each limit as a percentage of the best provider's for this model, averaged over the limits this provider was measured on. 100 = matches the best provider on every limit.";
export const OVERALL_METRIC: Metric = {
  key: OVERALL, tab: "Overall",
  question: "Which provider gives this model the most room overall?",
  explain: "Each limit as a percentage of the best provider's for this model, averaged over the limits that provider was measured on (token counts left out). 100 means it matches the best provider on every limit. Providers measured on fewer than two limits aren't scored.",
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
