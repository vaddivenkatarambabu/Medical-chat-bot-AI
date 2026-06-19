import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Loader2, Mail } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { AuthShell } from "@/components/auth/AuthShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/integrations/supabase/client";
import {
  authRedirect,
  isCurrentUserEmailVerified,
  normalizeEmail,
  requestEmailOtp,
  syncBackendAuthSession,
  validateEmail,
} from "@/lib/auth";

export const Route = createFileRoute("/sign-in")({
  head: () => ({
    meta: [
      {
        title: "Sign in - MediCore",
      },
      {
        name: "description",
        content: "Sign in to MediCore with your verified email and password.",
      },
    ],
  }),
  component: SignInPage,
});

function SignInPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [emailOtpLoading, setEmailOtpLoading] = useState(false);

  useEffect(() => {
    let active = true;

    const redirectIfSignedIn = async () => {
      try {
        const {
          data: { session },
        } = await supabase.auth.getSession();

        if (!active || !session?.user) {
          return;
        }

        if (await isCurrentUserEmailVerified()) {
          await syncBackendAuthSession();
          navigate({ to: "/chat", replace: true });
        }
      } catch {
        if (active) {
          toast.error("Authentication is not configured correctly.");
        }
      }
    };

    void redirectIfSignedIn();

    return () => {
      active = false;
    };
  }, [navigate]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (loading) return;

    const cleanEmail = normalizeEmail(email);
    const emailError = validateEmail(cleanEmail);
    if (emailError) {
      toast.error(emailError);
      return;
    }

    if (!password) {
      toast.error("Enter your password.");
      return;
    }

    setLoading(true);

    try {
      const { error } = await supabase.auth.signInWithPassword({
        email: cleanEmail,
        password,
      });

      if (error) {
        throw error;
      }

      if (!(await isCurrentUserEmailVerified())) {
        await supabase.auth.signOut();
        toast.error("Please verify your email before signing in.");
        navigate({
          to: "/verify-otp",
          search: {
            email: cleanEmail,
            purpose: "signup",
          },
        });
        return;
      }

      await syncBackendAuthSession();
      toast.success("Signed in successfully.");
      navigate({ to: "/chat", replace: true });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Sign in failed.");
    } finally {
      setLoading(false);
    }
  };

  const continueWithEmail = async () => {
    if (loading || emailOtpLoading) return;

    const cleanEmail = normalizeEmail(email);
    const emailError = validateEmail(cleanEmail);
    if (emailError) {
      toast.error(emailError);
      return;
    }

    setEmailOtpLoading(true);

    try {
      await requestEmailOtp(
        cleanEmail,
        authRedirect(
          `/verify-otp?email=${encodeURIComponent(cleanEmail)}&purpose=signin`,
        ),
        {
          createUser: false,
        },
      );

      toast.success("Verification code sent. Check your email.");
      navigate({
        to: "/verify-otp",
        search: {
          email: cleanEmail,
          purpose: "signin",
        },
      });
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not send verification.",
      );
    } finally {
      setEmailOtpLoading(false);
    }
  };

  return (
    <AuthShell
      title="Welcome back"
      subtitle="Sign in with your verified email and password."
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="signin-email">Email</Label>

          <Input
            id="signin-email"
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@email.com"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="signin-password">Password</Label>

          <Input
            id="signin-password"
            type="password"
            required
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="********"
          />
        </div>

        <Button type="submit" disabled={loading} className="h-11 w-full">
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Sign in"}
        </Button>
      </form>

      <Button
        type="button"
        variant="outline"
        disabled={loading || emailOtpLoading}
        className="mt-3 h-11 w-full"
        onClick={continueWithEmail}
      >
        {emailOtpLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <>
            <Mail className="h-4 w-4" />
            Continue with Email
          </>
        )}
      </Button>

      <div className="mt-5 flex items-center justify-between text-sm">
        <Link to="/forgot-password" className="text-primary hover:underline">
          Forgot password?
        </Link>

        <Link to="/sign-up" className="text-primary hover:underline">
          Create account
        </Link>
      </div>
    </AuthShell>
  );
}
