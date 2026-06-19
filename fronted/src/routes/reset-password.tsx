import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { AuthShell } from "@/components/auth/AuthShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/integrations/supabase/client";
import { syncBackendAuthSession, validatePassword } from "@/lib/auth";

type ResetMode = "create-password" | "reset-password";

type ResetSearch = {
  mode?: ResetMode;
};

export const Route = createFileRoute("/reset-password")({
  validateSearch: (search: Record<string, unknown>): ResetSearch => ({
    mode:
      search.mode === "create-password" ? "create-password" : "reset-password",
  }),
  head: () => ({
    meta: [
      {
        title: "Set password - MediCore",
      },
      {
        name: "description",
        content: "Create or reset your MediCore password.",
      },
    ],
  }),
  component: ResetPasswordPage,
});

function ResetPasswordPage() {
  const navigate = useNavigate();
  const { mode = "reset-password" } = Route.useSearch();
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionReady, setSessionReady] = useState(false);
  const [checkingSession, setCheckingSession] = useState(true);

  useEffect(() => {
    if (typeof window === "undefined") return;

    let active = true;

    const prepareSession = async () => {
      setCheckingSession(true);

      try {
        const params = new URLSearchParams(window.location.search);
        const code = params.get("code");

        if (code) {
          const { error } = await supabase.auth.exchangeCodeForSession(code);
          if (error) throw error;
        }

        const {
          data: { session },
        } = await supabase.auth.getSession();

        if (active) {
          setSessionReady(Boolean(session?.user));
        }
      } catch (error) {
        if (active) {
          setSessionReady(false);
          toast.error(
            error instanceof Error
              ? error.message
              : "Password reset link is invalid or expired.",
          );
        }
      } finally {
        if (active) {
          setCheckingSession(false);
        }
      }
    };

    void prepareSession();

    return () => {
      active = false;
    };
  }, []);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (loading) return;

    const passwordError = validatePassword(password);
    if (passwordError) {
      toast.error(passwordError);
      return;
    }

    if (password !== confirmPassword) {
      toast.error("Passwords do not match.");
      return;
    }

    setLoading(true);

    try {
      const { error } = await supabase.auth.updateUser({
        password,
      });

      if (error) {
        throw error;
      }

      await syncBackendAuthSession();
      toast.success(
        mode === "create-password"
          ? "Password created successfully."
          : "Password reset successfully.",
      );
      navigate({ to: "/chat", replace: true });
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not update password.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell
      title={
        mode === "create-password" ? "Create password" : "Set new password"
      }
      subtitle="Use a strong password that includes letters and numbers."
    >
      {checkingSession ? (
        <div className="flex items-center justify-center py-8 text-muted-foreground">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Checking secure session...
        </div>
      ) : !sessionReady ? (
        <div className="space-y-4 text-center">
          <p className="text-sm text-muted-foreground">
            This reset session is missing or expired. Request a new password
            reset email.
          </p>

          <Button asChild className="h-11 w-full">
            <Link to="/forgot-password">Send reset email</Link>
          </Button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="new-password">New password</Label>

            <Input
              id="new-password"
              type="password"
              required
              minLength={8}
              autoComplete="new-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="********"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="confirm-password">Confirm password</Label>

            <Input
              id="confirm-password"
              type="password"
              required
              minLength={8}
              autoComplete="new-password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              placeholder="********"
            />
          </div>

          <Button type="submit" disabled={loading} className="h-11 w-full">
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : mode === "create-password" ? (
              "Create password"
            ) : (
              "Reset password"
            )}
          </Button>
        </form>
      )}
    </AuthShell>
  );
}
