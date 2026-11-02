import rss from "@astrojs/rss";
import type { APIContext } from "astro";
import { changes, describeChange } from "../lib/data";

export function GET(context: APIContext) {
  return rss({
    title: "vlm.fyi changes",
    description: "Changes to measured request limits of vision-language model APIs.",
    site: context.site!,
    items: changes.map((c) => ({
      title: describeChange(c),
      pubDate: new Date(`${c.date}T00:00:00Z`),
      link: "/changes",
    })),
  });
}
