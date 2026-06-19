import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { AuthShell } from "@/components/auth/AuthShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/integrations/supabase/client";
import {
  authRedirect,
  normalizeEmail,
  requestEmailOtp,
  requestPasswordRecovery,
  syncBackendAuthSession,
  validateEmail,
} from "@/lib/auth";

type OtpPurpose = "signup" | "signin" | "recovery";

type VerifySearch = {
  email?: string;
  purpose?: OtpPurpose;
};

export const Route = createFileRoute("/verify-otp")({
  validateSearch: (search: Record<string, unknown>): VerifySearch => ({
    email: typeof search.email === "string" ? search.email : undefined,
    purpose:
      search.purpose === "recovery" || search.purpose === "signin"
        ? search.purpose
        : "signup",
  }),
  head: () => ({
    meta: [
      {
        title: "Verify email - MediCore",
      },
      {
        name: "description",
        content: "Verify your MediCore email code.",
      },
    ],
  }),
  component: VerifyOtpPage,
});

function VerifyOtpPage() {
  const navigate = useNavigate();
  const search = Route.useSearch();
  const purpose = search.purpose ?? "signup";

  const [email, setEmail] = useState(search.email ?? "");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [resending, setResending] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");
    if (!code) return;

    const verifyLink = async () => {
      setLoading(true);
      try {
        const { error } = await supabase.auth.exchangeCodeForSession(code);
        if (error) throw error;

        if (purpose === "signin") {
          await syncBackendAuthSession();
          toast.success("Signed in successfully.");
          navigate({
            to: "/chat",
            replace: true,
          });
          return;
        }

        toast.success("Email verified. Create your password.");
        navigate({
          to: "/reset-password",
          search: {
            mode: purpose === "signup" ? "create-password" : "reset-password",
          },
          replace: true,
        });
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Verification link failed.",
        );
      } finally {
        setLoading(false);
      }
    };

    void verifyLink();
  }, [navigate, purpose]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (loading) return;

    const cleanEmail = normalizeEmail(email);
    const emailError = validateEmail(cleanEmail);
    if (emailError) {
      toast.error(emailError);
      return;
    }

    const cleanToken = token.trim();
    if (cleanToken.length < 6) {
      toast.error("Enter the verification code from your email.");
      return;
    }

    setLoading(true);

    try {
      const { error } = await supabase.auth.verifyOtp({
        email: cleanEmail,
        token: cleanToken,
        type: purpose === "recovery" ? "recovery" : "email",
      });

      if (error) {
        throw error;
      }

      if (purpose === "signin") {
        await syncBackendAuthSession();
        toast.success("Signed in successfully.");
        navigate({
          to: "/chat",
          replace: true,
        });
        return;
      }

      toast.success("Email verified.");
      navigate({
        to: "/reset-password",
        search: {
          mode: purpose === "signup" ? "create-password" : "reset-password",
        },
        replace: true,
      });
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Verification failed.",
      );
    } finally {
      setLoading(false);
    }
  };

  const resendCode = async () => {
    if (resending) return;

    const cleanEmail = normalizeEmail(email);
    const emailError = validateEmail(cleanEmail);
    if (emailError) {
      toast.error(emailError);
      return;
    }

    setResending(true);

    try {
      if (purpose === "recovery") {
        await requestPasswordRecovery(
          cleanEmail,
          authRedirect("/reset-password"),
        );
      } else if (purpose === "signin") {
        await requestEmailOtp(
          cleanEmail,
          authRedirect(
            `/verify-otp?email=${encodeURIComponent(cleanEmail)}&purpose=signin`,
          ),
          {
            createUser: false,
          },
        );
      } else {
        await requestEmailOtp(
          cleanEmail,
          authRedirect(
            `/verify-otp?email=${encodeURIComponent(cleanEmail)}&purpose=signup`,
          ),
        );
      }

      toast.success("A new email has been sent.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not resend.");
    } finally {
      setResending(false);
    }
  };

  return (
    <AuthShell
      title={
        purpose === "recovery"
          ? "Verify reset code"
          : purpose === "signin"
            ? "Verify sign-in code"
            : "Verify your email"
      }
      subtitle="Enter the code sent to your email address."
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="verify-email">Email</Label>

          <Input
            id="verify-email"
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@email.com"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="verify-code">Verification code</Label>

          <Input
            id="verify-code"
            inputMode="numeric"
            autoComplete="one-time-code"
            required
            value={token}
            onChange={(event) => setToken(event.target.value)}
            placeholder="123456"
          />
        </div>

        <Button type="submit" disabled={loading} className="h-11 w-full">
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Verify"}
        </Button>
      </form>

      <div className="mt-5 flex items-center justify-between text-sm">
        <Button
          type="button"
          variant="ghost"
          disabled={resending}
          onClick={resendCode}
          className="px-0 text-primary hover:text-primary"
        >
          {resending ? "Sending..." : "Resend code"}
        </Button>

        <Link to="/sign-in" className="text-primary hover:underline">
          Sign in
        </Link>
      </div>
    </AuthShell>
  );
}
