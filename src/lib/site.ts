/** Public repository for the data and the site. */
export const REPO = "https://github.com/ambient-intelligence-hq/vlm-fyi";

/** Ambient, the team and product behind vlm.fyi. */
export const AMBIENT_REPO = "https://github.com/ambient-intelligence-hq/ambient";
export const CONTACT = "hello@vlm.fyi";

/** A link to the Ambient site tagged with where on vlm.fyi it was clicked, so each
 *  placement's clicks show up separately in Ambient's analytics. */
export function ambient(placement: string, path = ""): string {
  const u = new URL(path, "https://ambient-intelligence-hq.github.io/");
  u.search = new URLSearchParams({ utm_source: "vlm.fyi", utm_medium: "referral", utm_campaign: placement }).toString();
  return u.toString();
}
