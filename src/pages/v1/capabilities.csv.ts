import { entries, COLUMNS } from "../../lib/data";

const esc = (v: unknown) => {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function GET() {
  const head = ["id", "model_id", "model_name", "provider", "via", "api_model_id", "measured_at",
    ...COLUMNS.flatMap((c) => [`${c.key}_status`, `${c.key}_value`]), "accepts_video", "strict_tool_message_order"];
  const rows = entries.map((e) => [
    e.id, e.model.id, e.model.name, e.provider.name, e.provider.via ?? "", e.model.api_id, e.measured_at,
    ...COLUMNS.flatMap((c) => { const l = e.limits[c.key]; return [l?.status ?? "not_tested", l?.value ?? l?.at_least ?? ""]; }),
    e.accepts.video_url ?? "", e.behaviour.strict_tool_message_order ?? "",
  ]);
  return new Response([head, ...rows].map((r) => r.map(esc).join(",")).join("\n") + "\n", {
    headers: { "Content-Type": "text/csv; charset=utf-8", "Access-Control-Allow-Origin": "*" },
  });
}
