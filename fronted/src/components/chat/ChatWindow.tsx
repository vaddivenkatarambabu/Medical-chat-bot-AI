import { useEffect, useRef, useState, useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { SendHorizonal, Square, Mic, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { MessageBubble, TypingBubble } from "./MessageBubble";
import { MedicalDisclaimer } from "./MedicalDisclaimer";
import { SuggestionCards } from "./SuggestionCards";
import {
  ensureGuestSessionId,
  getAccessToken,
  getMessages,
  type ChatMessage,
} from "@/lib/conversations.functions";
import { backendUrl, getBackendUrl, readApiError } from "@/lib/api";
import { Logo } from "@/components/brand/Logo";
import { toast } from "sonner";
import { nanoid } from "nanoid";

type ChatStatus = "idle" | "submitted" | "streaming" | "error";

const MAX_MESSAGE_LENGTH = 2000;
const REQUEST_TIMEOUT_MS = 60000;

function createTextMessage(
  role: "user" | "assistant",
  text: string,
): ChatMessage {
  return {
    id: nanoid(),
    role,
    parts: [
      {
        type: "text",
        text,
      },
    ],
  };
}

export function ChatWindow({ conversationId }: { conversationId: string }) {
  const isGuest = conversationId === "guest";
  const queryClient = useQueryClient();

  const { data: initialMessages, isLoading } = useQuery({
    queryKey: ["messages", conversationId],
    queryFn: () => getMessages(conversationId),
    enabled: !isGuest,
  });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [input, setInput] = useState("");

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const userScrolledUpRef = useRef(false);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    abortRef.current?.abort();
    setMessages([]);
    setStatus("idle");
    setInput("");
    userScrolledUpRef.current = false;
    inputRef.current?.focus();
  }, [conversationId]);

  useEffect(() => {
    if (initialMessages && !isGuest) {
      setMessages(initialMessages);
    }
  }, [initialMessages, isGuest]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || userScrolledUpRef.current) return;

    el.scrollTo({
      top: el.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, status]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const handleScroll = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;

      userScrolledUpRef.current = !atBottom;
    };

    el.addEventListener("scroll", handleScroll, {
      passive: true,
    });

    return () => {
      el.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const isBusy = status === "submitted" || status === "streaming";

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
  }, []);

  const submit = useCallback(
    async (text?: string) => {
      const value = (text ?? input).trim();

      if (!value || isBusy) return;

      if (value.length > MAX_MESSAGE_LENGTH) {
        toast.error(`Message exceeds ${MAX_MESSAGE_LENGTH} characters`);
        return;
      }

      if (!getBackendUrl()) {
        toast.error("Backend URL is not configured");
        return;
      }

      const controller = new AbortController();
      let userMessage: ChatMessage | null = null;
      let timeoutId: number | undefined;
      abortRef.current = controller;

      try {
        const token = await getAccessToken();
        const guestSessionId = token ? undefined : ensureGuestSessionId();

        userMessage = createTextMessage("user", value);

        setMessages((prev) => [...prev, userMessage as ChatMessage]);

        setInput("");
        setStatus("submitted");
        userScrolledUpRef.current = false;

        timeoutId = window.setTimeout(() => {
          controller.abort();
        }, REQUEST_TIMEOUT_MS);

        const headers = new Headers({
          "Content-Type": "application/json",
        });

        if (token) {
          headers.set("Authorization", `Bearer ${token}`);
        } else if (guestSessionId) {
          headers.set("X-Guest-Session-Id", guestSessionId);
        }

        const res = await fetch(backendUrl("/get"), {
          method: "POST",
          headers,
          body: JSON.stringify({
            message: value,
            conversation_id: conversationId,
            client_message_id: userMessage.id,
            ...(guestSessionId
              ? {
                  guest_session_id: guestSessionId,
                }
              : {}),
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          throw new Error(await readApiError(res));
        }

        if (!mountedRef.current) {
          return;
        }

        setStatus("streaming");

        const contentType = res.headers.get("content-type") ?? "";

        let answerText = "";

        if (contentType.includes("application/json")) {
          const data = await res.json();

          const answer = data?.answer ?? data?.message ?? "";
          answerText = typeof answer === "string" ? answer : String(answer);
        } else {
          answerText = await res.text();
        }

        const safeAnswer =
          answerText.trim() || "Sorry, I couldn't generate a response.";

        const assistantMessage = createTextMessage("assistant", safeAnswer);

        setMessages((prev) => [...prev, assistantMessage]);

        setStatus("idle");

        void Promise.all([
          ...(conversationId === "guest"
            ? []
            : [
                queryClient.invalidateQueries({
                  queryKey: ["messages", conversationId],
                }),
              ]),
          queryClient.invalidateQueries({
            queryKey: ["conversations"],
          }),
        ]);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          if (mountedRef.current) {
            setStatus("idle");
          }
          return;
        }

        if (!mountedRef.current) {
          return;
        }

        setStatus("error");

        const message =
          err instanceof Error ? err.message : "Something went wrong";

        toast.error(message);

        if (userMessage) {
          const failedMessageId = userMessage.id;
          setMessages((prev) => prev.filter((m) => m.id !== failedMessageId));
        }
      } finally {
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
        }
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        if (mountedRef.current) {
          inputRef.current?.focus();
        }
      }
    },
    [conversationId, input, isBusy, queryClient],
  );

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const empty = messages.length === 0;

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 md:px-8 py-6">
        <div className="mx-auto max-w-3xl space-y-6">
          {isLoading && (
            <div className="text-sm text-muted-foreground">
              Loading conversation...
            </div>
          )}

          {empty && !isLoading && (
            <div className="flex flex-col items-center text-center pt-8 pb-4">
              <Logo size={48} withText={false} />

              <h1 className="mt-5 font-display text-3xl font-bold">
                How can I help you today?
              </h1>

              <p className="mt-2 text-sm text-muted-foreground max-w-md">
                Describe your symptoms or ask any health question. I'll help
                guide you through it.
              </p>

              <div className="mt-8 w-full">
                <SuggestionCards onPick={(p) => submit(p)} />
              </div>
            </div>
          )}

          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}

          {(status === "submitted" || status === "streaming") && (
            <TypingBubble />
          )}
        </div>
      </div>

      <div className="border-t bg-background/80 backdrop-blur sticky bottom-0">
        <div className="mx-auto max-w-3xl px-4 md:px-8 py-4 space-y-3">
          <MedicalDisclaimer />

          <div className="relative rounded-2xl border bg-card shadow-[var(--shadow-elegant)] focus-within:ring-2 focus-within:ring-primary/50 transition">
            <Textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Describe symptoms, ask a health question..."
              rows={1}
              className="min-h-[56px] max-h-40 resize-none border-0 bg-transparent pr-28 focus-visible:ring-0 focus-visible:ring-offset-0"
            />

            <div className="absolute right-2 bottom-2 flex items-center gap-1">
              <Button
                size="icon-sm"
                variant="ghost"
                onClick={() => toast.info("Voice input coming soon")}
                aria-label="Voice input"
              >
                <Mic className="h-4 w-4" />
              </Button>

              {isBusy ? (
                <Button
                  size="icon-sm"
                  variant="destructive"
                  onClick={stop}
                  aria-label="Stop generating"
                >
                  <Square className="h-3.5 w-3.5 fill-current" />
                </Button>
              ) : (
                <Button
                  size="icon-sm"
                  onClick={() => submit()}
                  disabled={!input.trim() || isBusy}
                  aria-label="Send message"
                  className="shadow-[var(--shadow-elegant)]"
                >
                  <SendHorizonal className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>

          <p className="text-[10px] text-center text-muted-foreground flex items-center justify-center gap-1">
            <Sparkles className="h-3 w-3" />
            MediCore can make mistakes. Verify important medical information.
          </p>
        </div>
      </div>
    </div>
  );
}
