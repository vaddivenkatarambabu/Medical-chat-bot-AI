import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Plus, Search, Trash2, LogOut, MessageSquare, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Logo } from "@/components/brand/Logo";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  getAccessToken,
  listConversations,
  createConversation,
  deleteConversation,
} from "@/lib/conversations.functions";
import { supabase } from "@/integrations/supabase/client";
import { cn } from "@/lib/utils";
import { revokeBackendAuthSession } from "@/lib/auth";

export function ChatSidebar({ onClose }: { onClose?: () => void }) {
  const navigate = useNavigate();
  const qc = useQueryClient();

  // strict: false returns a partial params object — type it properly instead of casting
  const { conversationId } =
    (useParams({ strict: false }) as { conversationId?: string }) ?? {};

  const [search, setSearch] = useState("");
  const [hasSession, setHasSession] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);

  const { data: conversations = [] } = useQuery({
    queryKey: ["conversations"],
    queryFn: listConversations,
    enabled: authChecked,
  });

  const createMut = useMutation({
    mutationFn: () => createConversation(),
    onSuccess: (c) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      navigate({
        to: "/chat/$conversationId",
        params: { conversationId: c.id },
      });
      onClose?.();
    },
    onError: () => {
      toast.error("Could not create saved conversation. Opening guest chat.");
      navigate({
        to: "/chat/$conversationId",
        params: {
          conversationId: "guest",
        },
      });
      onClose?.();
    },
  });

  const deleteMut = useMutation({
    mutationFn: deleteConversation,
    onSuccess: (_, id) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      if (conversationId === id) navigate({ to: "/chat" });
      toast.success("Conversation deleted");
    },
    onError: () => toast.error("Could not delete conversation"),
  });

  useEffect(() => {
    let active = true;
    let unsubscribe: (() => void) | undefined;

    const refreshSessionState = async () => {
      const token = await getAccessToken();

      if (!active) {
        return;
      }

      setHasSession(Boolean(token));
      setAuthChecked(true);
    };

    void refreshSessionState();

    try {
      const {
        data: { subscription },
      } = supabase.auth.onAuthStateChange((_event, session) => {
        const signedIn = Boolean(session?.access_token);
        setHasSession(signedIn);
        setAuthChecked(true);

        if (!signedIn) {
          qc.removeQueries({ queryKey: ["messages"] });
        }

        void qc.invalidateQueries({ queryKey: ["conversations"] });
      });

      unsubscribe = () => subscription.unsubscribe();
    } catch {
      if (active) {
        setAuthChecked(true);
      }
    }

    return () => {
      active = false;
      unsubscribe?.();
    };
  }, [qc]);

  const openGuestChat = () => {
    navigate({
      to: "/chat/$conversationId",
      params: {
        conversationId: "guest",
      },
    });
    onClose?.();
  };

  const startConversation = () => {
    if (!hasSession) {
      openGuestChat();
      return;
    }

    createMut.mutate();
  };

  const signOut = async () => {
    if (!hasSession) {
      navigate({ to: "/sign-in" });
      onClose?.();
      return;
    }

    try {
      await qc.cancelQueries();
      await revokeBackendAuthSession();
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      qc.clear();
      navigate({ to: "/sign-in", replace: true });
    } catch {
      toast.error("Could not sign out. Please try again.");
    }
  };

  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <aside className="flex h-full w-full md:w-72 flex-col bg-sidebar border-r">
      <div className="flex items-center justify-between p-4">
        <Link to="/" className="flex items-center">
          <Logo size={28} />
        </Link>
        {onClose && (
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={onClose}
            aria-label="Close sidebar"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      <div className="px-3">
        <Button
          onClick={startConversation}
          disabled={createMut.isPending}
          className="w-full h-10 shadow-[var(--shadow-elegant)]"
        >
          <Plus className="h-4 w-4 mr-2" /> New consultation
        </Button>
      </div>

      <div className="px-3 mt-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search conversations"
            className="pl-8 h-9 bg-background/50"
          />
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto mt-3 px-2 space-y-1">
        {filtered.length === 0 && (
          <p className="px-3 py-6 text-xs text-muted-foreground text-center">
            No conversations yet
          </p>
        )}
        {filtered.map((c) => {
          const active = conversationId === c.id;
          const isDeleting =
            deleteMut.isPending && deleteMut.variables === c.id;

          return (
            <div
              key={c.id}
              className={cn(
                "group flex items-center gap-2 rounded-lg px-2 py-1.5",
                active ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/50",
              )}
            >
              <Link
                to="/chat/$conversationId"
                params={{ conversationId: c.id }}
                onClick={onClose}
                className="flex-1 flex items-center gap-2 min-w-0"
              >
                <MessageSquare className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <span className="text-sm truncate">{c.title}</span>
              </Link>

              <Button
                size="icon-sm"
                variant="ghost"
                disabled={isDeleting}
                onClick={(e) => {
                  e.stopPropagation();
                  deleteMut.mutate(c.id);
                }}
                aria-label="Delete conversation"
                className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          );
        })}
      </nav>

      <div className="border-t p-3 flex items-center justify-between">
        <ThemeToggle />
        <Button size="sm" variant="ghost" onClick={signOut}>
          <LogOut className="h-4 w-4 mr-1.5" />{" "}
          {hasSession ? "Sign out" : "Sign in"}
        </Button>
      </div>
    </aside>
  );
}
