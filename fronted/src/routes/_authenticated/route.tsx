import { createFileRoute, Outlet } from "@tanstack/react-router";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/_authenticated")({
  ssr: false,

  beforeLoad: async () => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      return {
        user: session?.user ?? null,
      };
    } catch (error) {
      console.warn("Authenticated route session check failed", {
        message: error instanceof Error ? error.message : "unknown error",
      });
      return {
        user: null,
        authError: true,
      };
    }
  },

  component: () => <Outlet />,
});
