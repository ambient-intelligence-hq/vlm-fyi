import changes from "../../../data/changes.json";

export function GET() {
  return new Response(JSON.stringify(changes, null, 1), {
    headers: { "Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*" },
  });
}
