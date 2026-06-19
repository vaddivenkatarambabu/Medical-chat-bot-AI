// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - tanstackStart, viteReact, tailwindcss, tsConfigPaths, nitro (build-only using cloudflare as a default target),
//     componentTagger (dev-only), VITE_* env injection, @ path alias, React/TanStack dedupe,
//     error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

export default defineConfig({
  tanstackStart: {
    // Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
    // nitro/vite builds from this
    server: { entry: "server" },
  },
  vite: {
    build: {
      rollupOptions: {
        output: {
          manualChunks(id: string) {
            const normalizedId = id.replaceAll("\\", "/");

            if (!normalizedId.includes("node_modules")) {
              return;
            }

            if (
              normalizedId.includes("/react/") ||
              normalizedId.includes("/react-dom/") ||
              normalizedId.includes("@tanstack/react-router") ||
              normalizedId.includes("@tanstack/react-query")
            ) {
              return "react-core";
            }

            if (normalizedId.includes("@supabase")) {
              return "supabase";
            }

            if (normalizedId.includes("framer-motion")) {
              return "motion";
            }

            if (normalizedId.includes("lucide-react")) {
              return "icons";
            }

            if (
              normalizedId.includes("react-markdown") ||
              normalizedId.includes("remark-") ||
              normalizedId.includes("micromark") ||
              normalizedId.includes("unified") ||
              normalizedId.includes("hast") ||
              normalizedId.includes("mdast")
            ) {
              return "markdown";
            }

            return;
          },
        },
      },
    },
  },
});
