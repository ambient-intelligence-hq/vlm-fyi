import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://vlm.fyi",
  trailingSlash: "ignore",
  build: { format: "directory" },
});
