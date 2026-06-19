import { supabase } from "@/integrations/supabase/client";
import { backendUrl, getBackendUrl, readApiError } from "@/lib/api";

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

export function validateEmail(email: string): string | null {
  if (!EMAIL_PATTERN.test(normalizeEmail(email))) {
    return "Enter a valid email address.";
  }

  return null;
}

export function validatePassword(password: string): string | null {
  if (password.length < 8) {
    return "Password must be at least 8 characters.";
  }

  if (!/[A-Za-z]/.test(password) || !/[0-9]/.test(password)) {
    return "Password must include letters and numbers.";
  }

  return null;
}

export function authRedirect(path: string): string {
  if (typeof window === "undefined") {
    return path;
  }

  return new URL(path, window.location.origin).toString();
}

async function postAuthEmail(
  path: string,
  email: string,
  redirectTo: string,
  options: { createUser?: boolean } = {},
) {
  if (!getBackendUrl()) {
    throw new Error("Backend URL is not configured.");
  }

  const response = await fetch(backendUrl(path), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email: normalizeEmail(email),
      redirect_to: redirectTo,
      ...(options.createUser === undefined
        ? {}
        : {
            create_user: options.createUser,
          }),
    }),
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }
}

export async function requestEmailOtp(
  email: string,
  redirectTo: string,
  options: { createUser?: boolean } = {},
): Promise<void> {
  await postAuthEmail("/api/auth/send-otp", email, redirectTo, options);
}

export async function requestPasswordRecovery(
  email: string,
  redirectTo: string,
): Promise<void> {
  await postAuthEmail("/api/auth/send-recovery", email, redirectTo);
}

export async function isCurrentUserEmailVerified(): Promise<boolean> {
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();

  if (error || !user) {
    return false;
  }

  return Boolean(user.email_confirmed_at);
}

export async function syncBackendAuthSession(): Promise<void> {
  if (!getBackendUrl()) {
    return;
  }

  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    return;
  }

  await fetch(backendUrl("/api/auth/session"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${session.access_token}`,
    },
  }).catch(() => undefined);
}

export async function revokeBackendAuthSession(): Promise<void> {
  if (!getBackendUrl()) {
    return;
  }

  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    return;
  }

  await fetch(backendUrl("/api/auth/logout"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${session.access_token}`,
    },
  }).catch(() => undefined);
}
