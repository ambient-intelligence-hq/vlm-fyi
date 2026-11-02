import dataset from "../../../data/capabilities.json";

export function GET() {
  return new Response(JSON.stringify(dataset, null, 1), {
    headers: { "Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*" },
  });
}
