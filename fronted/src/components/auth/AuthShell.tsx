import { Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import type { ReactNode } from "react";
import { Logo } from "@/components/brand/Logo";
import { Blobs } from "@/components/landing/Blobs";

export function AuthShell({
  title,
  subtitle,
  children,
  footer,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className="relative flex min-h-dvh items-center justify-center overflow-hidden bg-background px-4">
      <Blobs />

      <div
        className="absolute inset-0"
        style={{
          background: "var(--gradient-hero)",
        }}
        aria-hidden
      />

      <Link
        to="/"
        className="absolute left-6 top-6 z-10 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </Link>

      <motion.div
        initial={{
          opacity: 0,
          y: 20,
        }}
        animate={{
          opacity: 1,
          y: 0,
        }}
        className="glass-panel relative z-10 w-full max-w-md rounded-3xl p-8 shadow-[var(--shadow-elegant)]"
      >
        <div className="flex flex-col items-center text-center">
          <Logo size={40} />

          <h1 className="mt-6 font-display text-2xl font-bold">{title}</h1>

          <p className="mt-2 text-sm text-muted-foreground">{subtitle}</p>
        </div>

        <div className="mt-8">{children}</div>

        {footer ?? (
          <p className="mt-6 text-center text-xs text-muted-foreground">
            By continuing you agree this is not a substitute for professional
            medical advice.
          </p>
        )}
      </motion.div>
    </div>
  );
}
