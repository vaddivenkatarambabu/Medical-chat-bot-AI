import { supabase } from "@/integrations/supabase/client";
import { backendUrl, getBackendUrl, readApiError } from "@/lib/api";

const LEGACY_GUEST_SESSION_STORAGE_KEY = "medicore_guest_session_id";
let guestSessionInitialization: Promise<void> | null = null;

const textPartSchema = {
  isValid(value: unknown): value is ChatMessagePart {
    return (
      typeof value === "object" &&
      value !== null &&
      "type" in value &&
      value.type === "text" &&
      "text" in value &&
      typeof value.text === "string"
    );
  },
};

type ChatMessagePart = {
  type: "text";
  text: string;
  state?: "streaming" | "done";
};

export type ChatSource = {
  source: string;
  page?: number | string | null;
  page_label?: string | null;
  chunk?: number | string | null;
  knowledge_base_version?: string | null;
};

export type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant";
  parts: ChatMessagePart[];
  sources?: ChatSource[];
};

export type ConversationSummary = {
  id: string;
  external_id: string | null;
  title: string;
  summary: string | null;
  created_at: string;
  updated_at: string;
};

type BackendMessage = {
  id: string;
  role: string;
  content?: string | null;
  parts?: unknown;
  client_message_id?: string | null;
};

type GuestSessionResponse = {
  ok?: boolean;
  expires_at?: string;
};

async function requestGuestSession(): Promise<void> {
  if (!getBackendUrl()) {
    throw new Error("Backend URL is not configured.");
  }

  const response = await fetch(backendUrl("/api/guest-session"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    credentials: "include",
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  const data = (await response
    .json()
    .catch(() => ({}))) as GuestSessionResponse;

  if (data.ok !== true) {
    throw new Error("Guest session initialization failed.");
  }
}

export async function ensureGuestSession(): Promise<void> {
  if (typeof window !== "undefined") {
    try {
      window.localStorage.removeItem(LEGACY_GUEST_SESSION_STORAGE_KEY);
    } catch {
      // Ignore storage cleanup failures. The HttpOnly cookie remains authoritative.
    }
  }

  if (!guestSessionInitialization) {
    guestSessionInitialization = requestGuestSession().catch((error) => {
      guestSessionInitialization = null;
      throw error;
    });
  }

  await guestSessionInitialization;
}

function normalizeParts(
  parts: unknown,
  content?: string | null,
): ChatMessagePart[] {
  if (Array.isArray(parts)) {
    const valid = parts.filter(textPartSchema.isValid);

    if (valid.length > 0) {
      return valid;
    }
  }

  return [
    {
      type: "text",
      text: content ?? "",
    },
  ];
}

function mapBackendMessage(message: BackendMessage): ChatMessage | null {
  if (
    message.role !== "system" &&
    message.role !== "user" &&
    message.role !== "assistant"
  ) {
    return null;
  }

  return {
    id: message.client_message_id || message.id,
    role: message.role,
    parts: normalizeParts(message.parts, message.content),
  };
}

export async function getAccessToken(): Promise<string | null> {
  try {
    const {
      data: { session },
    } = await supabase.auth.getSession();

    return session?.access_token ?? null;
  } catch {
    return null;
  }
}

async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
  tokenOverride?: string | null,
): Promise<T> {
  if (!getBackendUrl()) {
    throw new Error("Backend URL is not configured.");
  }

  const token = tokenOverride ?? (await getAccessToken());
  const headers = new Headers(options.headers);

  headers.set("Accept", "application/json");

  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(backendUrl(path), {
    ...options,
    headers,
    credentials: "include",
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const token = await getAccessToken();

  if (!token) {
    await ensureGuestSession();
  }

  return apiRequest<ConversationSummary[]>("/api/conversations", {}, token);
}

export async function createConversation(data: { title?: string } = {}) {
  const token = await getAccessToken();

  if (!token) {
    await ensureGuestSession();
  }

  return apiRequest<ConversationSummary>(
    "/api/conversations",
    {
      method: "POST",
      body: JSON.stringify(data),
    },
    token,
  );
}

export async function deleteConversation(id: string): Promise<{ ok: true }> {
  const token = await getAccessToken();

  if (!token) {
    await ensureGuestSession();
  }

  return apiRequest<{ ok: true }>(
    `/api/conversations/${encodeURIComponent(id)}`,
    {
      method: "DELETE",
    },
    token,
  );
}

export async function getMessages(
  conversationId: string,
): Promise<ChatMessage[]> {
  const token = await getAccessToken();

  if (!token) {
    await ensureGuestSession();
  }

  const messages = await apiRequest<BackendMessage[]>(
    `/api/conversations/${encodeURIComponent(conversationId)}/messages`,
    {},
    token,
  );

  return messages
    .map(mapBackendMessage)
    .filter((message): message is ChatMessage => message !== null);
}
