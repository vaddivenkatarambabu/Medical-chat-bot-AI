import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, Check, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import logo from "@/assets/medicore-logo.png";
import type { ChatMessage } from "@/lib/conversations.functions";

// Extracted as a constant to avoid creating a new object on every render
const BUBBLE_TRANSITION = { duration: 0.25 };

// Typing indicator dot delays in seconds
const TYPING_DOT_DELAYS = [0, 0.15, 0.3];

function getText(m: ChatMessage): string {
  return m.parts.map((p) => (p.type === "text" ? p.text : "")).join("");
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const copyTimeoutRef = useRef<number | undefined>(undefined);
  const text = getText(message);

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) {
        window.clearTimeout(copyTimeoutRef.current);
      }
    };
  }, []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      toast.success("Copied to clipboard");
      if (copyTimeoutRef.current) {
        window.clearTimeout(copyTimeoutRef.current);
      }
      copyTimeoutRef.current = window.setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Could not copy to clipboard");
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={BUBBLE_TRANSITION}
      className={cn(
        "group flex gap-3 w-full",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      {!isUser && (
        <img
          src={logo}
          alt=""
          width={32}
          height={32}
          className="h-8 w-8 rounded-lg shrink-0"
        />
      )}
      <div className={cn("max-w-[85%] md:max-w-[75%]", isUser && "order-1")}>
        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm leading-relaxed",
            isUser
              ? "bg-primary text-primary-foreground rounded-br-md"
              : "bg-card border rounded-bl-md",
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{text}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-2 prose-headings:mt-3 prose-headings:mb-1 prose-ul:my-2 prose-code:bg-muted prose-code:px-1 prose-code:rounded">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {text || "..."}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && text && (
          <div className="opacity-0 group-hover:opacity-100 transition mt-1 flex">
            <Button
              size="icon-sm"
              variant="ghost"
              onClick={copy}
              aria-label="Copy message"
            >
              {copied ? (
                <Check className="h-3.5 w-3.5" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
            </Button>
          </div>
        )}
      </div>
      {isUser && (
        <div className="h-8 w-8 shrink-0 rounded-lg bg-accent text-accent-foreground flex items-center justify-center">
          <User className="h-4 w-4" />
        </div>
      )}
    </motion.div>
  );
}

export function TypingBubble() {
  return (
    <div className="flex gap-3">
      <img
        src={logo}
        alt=""
        width={32}
        height={32}
        className="h-8 w-8 rounded-lg"
      />
      <div className="rounded-2xl rounded-bl-md bg-card border px-4 py-3 flex items-center gap-1.5">
        {TYPING_DOT_DELAYS.map((delay, index) => (
          <span
            key={index}
            className="h-1.5 w-1.5 rounded-full bg-primary"
            style={{
              animation: "typing-dot 1.2s infinite",
              animationDelay: `${delay}s`,
            }}
          />
        ))}
      </div>
    </div>
  );
}
