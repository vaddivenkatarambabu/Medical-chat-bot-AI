import { supabase } from "@/integrations/supabase/client";
import { backendUrl, getBackendUrl, readApiError } from "@/lib/api";
import { nanoid } from "nanoid";

const GUEST_SESSION_STORAGE_KEY = "medicore_guest_session_id";

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

export type ChatMessagePart = {
  type: "text";
  text: string;
  state?: "streaming" | "done";
};

export type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant";
  parts: ChatMessagePart[];
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

export function getGuestSessionId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    return window.localStorage.getItem(GUEST_SESSION_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function ensureGuestSessionId(): string {
  if (typeof window === "undefined") {
    return nanoid();
  }

  const existing = getGuestSessionId();
  if (existing) {
    return existing;
  }

  const created = nanoid();

  try {
    window.localStorage.setItem(GUEST_SESSION_STORAGE_KEY, created);
  } catch {
    return created;
  }

  return created;
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
  identity: {
    token?: string | null;
    guestSessionId?: string | null;
  } = {},
): Promise<T> {
  if (!getBackendUrl()) {
    throw new Error("Backend URL is not configured.");
  }

  const token = identity.token ?? (await getAccessToken());
  const guestSessionId =
    identity.guestSessionId ?? (!token ? getGuestSessionId() : null);
  const headers = new Headers(options.headers);

  headers.set("Accept", "application/json");
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  } else if (guestSessionId) {
    headers.set("X-Guest-Session-Id", guestSessionId);
  }

  const response = await fetch(backendUrl(path), {
    ...options,
    headers,
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

function guestQuery(
  token: string | null,
  guestSessionId: string | null,
): string {
  return !token && guestSessionId
    ? `?guest_session_id=${encodeURIComponent(guestSessionId)}`
    : "";
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const token = await getAccessToken();
  const guestSessionId = token ? null : getGuestSessionId();

  if (!token && !guestSessionId) {
    return [];
  }

  const query = guestQuery(token, guestSessionId);

  return apiRequest<ConversationSummary[]>(
    `/api/conversations${query}`,
    {},
    {
      token,
      guestSessionId,
    },
  );
}

export async function createConversation(data: { title?: string } = {}) {
  const token = await getAccessToken();
  const guestSessionId = token ? null : ensureGuestSessionId();

  const body = {
    ...data,
    ...(!token && guestSessionId
      ? {
          guest_session_id: guestSessionId,
        }
      : {}),
  };

  return apiRequest<ConversationSummary>(
    "/api/conversations",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
    {
      token,
      guestSessionId,
    },
  );
}

export async function renameConversation(data: {
  id: string;
  title: string;
}): Promise<ConversationSummary> {
  const token = await getAccessToken();
  const guestSessionId = token ? null : getGuestSessionId();
  const query = guestQuery(token, guestSessionId);

  return apiRequest<ConversationSummary>(
    `/api/conversations/${encodeURIComponent(data.id)}${query}`,
    {
      method: "PATCH",
      body: JSON.stringify({
        title: data.title,
        ...(!token && guestSessionId
          ? {
              guest_session_id: guestSessionId,
            }
          : {}),
      }),
    },
    {
      token,
      guestSessionId,
    },
  );
}

export async function deleteConversation(id: string): Promise<{ ok: true }> {
  const token = await getAccessToken();
  const guestSessionId = token ? null : getGuestSessionId();
  const query = guestQuery(token, guestSessionId);

  return apiRequest<{ ok: true }>(
    `/api/conversations/${encodeURIComponent(id)}${query}`,
    {
      method: "DELETE",
    },
    {
      token,
      guestSessionId,
    },
  );
}

export async function getMessages(
  conversationId: string,
): Promise<ChatMessage[]> {
  const token = await getAccessToken();
  const guestSessionId = token ? null : getGuestSessionId();

  if (!token && !guestSessionId) {
    return [];
  }

  const query = guestQuery(token, guestSessionId);

  const messages = await apiRequest<BackendMessage[]>(
    `/api/conversations/${encodeURIComponent(conversationId)}/messages${query}`,
    {},
    {
      token,
      guestSessionId,
    },
  );

  return messages
    .map(mapBackendMessage)
    .filter((message): message is ChatMessage => message !== null);
}
