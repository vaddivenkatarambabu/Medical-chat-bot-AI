import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { AuthShell } from "@/components/auth/AuthShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  authRedirect,
  normalizeEmail,
  requestPasswordRecovery,
  validateEmail,
} from "@/lib/auth";

export const Route = createFileRoute("/forgot-password")({
  head: () => ({
    meta: [
      {
        title: "Forgot password - MediCore",
      },
      {
        name: "description",
        content: "Reset your MediCore account password.",
      },
    ],
  }),
  component: ForgotPasswordPage,
});

function ForgotPasswordPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (loading) return;

    const cleanEmail = normalizeEmail(email);
    const emailError = validateEmail(cleanEmail);
    if (emailError) {
      toast.error(emailError);
      return;
    }

    setLoading(true);

    try {
      await requestPasswordRecovery(
        cleanEmail,
        authRedirect("/reset-password"),
      );

      toast.success("Password reset email sent.");
      navigate({
        to: "/verify-otp",
        search: {
          email: cleanEmail,
          purpose: "recovery",
        },
      });
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not send reset email.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell
      title="Reset your password"
      subtitle="Enter your email and we will send a secure reset email."
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="forgot-email">Email</Label>

          <Input
            id="forgot-email"
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@email.com"
          />
        </div>

        <Button type="submit" disabled={loading} className="h-11 w-full">
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            "Send reset email"
          )}
        </Button>
      </form>

      <p className="mt-5 text-center text-sm text-muted-foreground">
        Remembered your password?{" "}
        <Link to="/sign-in" className="text-primary hover:underline">
          Sign in
        </Link>
      </p>
    </AuthShell>
  );
}
