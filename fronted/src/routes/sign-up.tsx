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
  requestEmailOtp,
  validateEmail,
} from "@/lib/auth";

export const Route = createFileRoute("/sign-up")({
  head: () => ({
    meta: [
      {
        title: "Create account - MediCore",
      },
      {
        name: "description",
        content: "Create a verified MediCore account.",
      },
    ],
  }),
  component: SignUpPage,
});

function SignUpPage() {
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
      await requestEmailOtp(
        cleanEmail,
        authRedirect(
          `/verify-otp?email=${encodeURIComponent(cleanEmail)}&purpose=signup`,
        ),
      );

      toast.success("Verification code sent. Check your email.");
      navigate({
        to: "/verify-otp",
        search: {
          email: cleanEmail,
          purpose: "signup",
        },
      });
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not send verification.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell
      title="Create your account"
      subtitle="Enter your email and we will send a verification code."
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="signup-email">Email</Label>

          <Input
            id="signup-email"
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
            "Send verification code"
          )}
        </Button>
      </form>

      <p className="mt-5 text-center text-sm text-muted-foreground">
        Already verified?{" "}
        <Link to="/sign-in" className="text-primary hover:underline">
          Sign in
        </Link>
      </p>
    </AuthShell>
  );
}
