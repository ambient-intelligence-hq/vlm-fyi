// Usage: npx tsx scripts/audit.mts
// Every gap behind the overall score: which limits each pair isn't scored on, why, and how far
// the score could move if that limit were measured (missing = 0% vs missing = 100%).
const m = await import("../src/lib/data.ts");
const K = m.RANKED as string[];
const short: Record<string, string> = { images_per_request: "images", frames_768x432: "frames", inline_request_bytes: "inline",
  url_image_bytes: "url", video_file_bytes: "vfile", video_tokens_per_min: "density" };
const reason = (e: any, k: string): string => {
  const l = e.limits[k];
  if (!l || l.status === "not_tested") {
    const via = e.provider.via === "openrouter";
    if (k === "video_file_bytes" && e.notes?.some((n: string) => n.includes("only inline"))) return "inline-only video (URL test can't run)";
    if (via && ["inline_request_bytes", "url_image_bytes", "video_file_bytes", "frames_768x432"].includes(k)) return "OpenRouter upstream: not in sweep";
    return "not tested";
  }
  if (l.status === "inconclusive") return `no verdict (${l.reason ?? l.failure ?? "?"})`;
  return `${l.status} but unscored`;
};
const rows: any[] = [];
for (const e of m.entries) {
  if (m.NO_OVERALL[m.modelSlug(e)]) continue;   // no overall score for this model (docs/TODO-reruns.md)
  const sh = K.map((k) => m.shareOfBest(k, e));
  const got = sh.filter((x: any) => x !== null) as number[];
  const miss = K.filter((_, i) => sh[i] === null);
  // a limit nobody can be compared on (no peer measured, or no provider takes it) isn't a gap
  const gaps = miss.filter((k) => m.peers(k, e).filter((b: any) => b.value !== null).length >= 1 && !(m.peers(k, e).length < 2));
  const sum = got.reduce((a, b) => a + b, 0);
  // Scored, but on a floor: the search stopped early or hit our cap below the best provider's value.
  const floors = K.filter((k, i) => sh[i] !== null && sh[i]! < 1 && ["at_least", "inconclusive"].includes(e.limits[k]?.status)
    && !(k === "video_file_bytes" && e.limits.video_tokens_per_min?.status === "ignored"));   // scored 0 on purpose
  const slack = floors.reduce((a, k) => a + (1 - m.shareOfBest(k, e)!), 0);
  const lo = Math.round(100 * sum / (got.length + gaps.length)), hi = Math.round(100 * (sum + slack + gaps.length) / (got.length + gaps.length));
  rows.push({ model: e.model.name, prov: m.providerLabel(e), score: m.overall(e)?.score ?? null, n: got.length, lo, hi,
    gaps: [...gaps.map((k) => `${short[k]}: ${reason(e, k)}`),
      ...floors.map((k) => `${short[k]}: floor, ${e.limits[k].status === "at_least" ? "passed our cap" : "search stopped early"} at ${Math.round(100 * m.shareOfBest(k, e)!)}% of best`)] });
}
for (const r of rows.filter((r) => r.gaps.length)) console.log(`${r.model} · ${r.prov}  score ${r.score ?? "—"} (could be ${r.lo}–${r.hi}), ${r.n}/6 limits\n   ${r.gaps.join("\n   ")}`);
console.log(`\n${rows.filter((r) => r.gaps.length).length} of ${rows.length} pairs have gaps`);
