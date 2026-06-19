import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { createConversation } from "@/lib/conversations.functions";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/_authenticated/chat/")({
  component: NewChatRedirect,
});

function NewChatRedirect() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;

    started.current = true;

    const createNewConversation = async () => {
      try {
        const {
          data: { session },
          error: sessionError,
        } = await supabase.auth.getSession();

        if (sessionError || !session?.user) {
          navigate({
            to: "/chat/$conversationId",
            params: {
              conversationId: "guest",
            },
            replace: true,
          });

          return;
        }

        const conversation = await createConversation();

        await qc.invalidateQueries({
          queryKey: ["conversations"],
        });

        navigate({
          to: "/chat/$conversationId",
          params: {
            conversationId: conversation.id,
          },
          replace: true,
        });
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Failed to start a new consultation";

        if (/unauthorized|invalid token|no authorization/i.test(message)) {
          toast.info("Opening a guest consultation");

          navigate({
            to: "/chat/$conversationId",
            params: {
              conversationId: "guest",
            },
            replace: true,
          });

          return;
        }

        toast.error(`${message}. Opening a guest consultation.`);

        navigate({
          to: "/chat/$conversationId",
          params: {
            conversationId: "guest",
          },
          replace: true,
        });
      }
    };

    void createNewConversation();
  }, [navigate, qc]);

  return (
    <div className="flex h-full items-center justify-center text-muted-foreground">
      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
      Starting consultation...
    </div>
  );
}
