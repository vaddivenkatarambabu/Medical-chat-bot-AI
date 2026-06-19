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
    } catch {
      return {
        user: null,
      };
    }
  },

  component: () => <Outlet />,
});
