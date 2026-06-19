const DEFAULT_DEV_BACKEND_URL = "http://127.0.0.1:1819";

const serverEnv =
  typeof process !== "undefined" && process.env ? process.env : {};

export const BACKEND_URL = (
  import.meta.env.VITE_BACKEND_URL ||
  serverEnv.VITE_BACKEND_URL ||
  serverEnv.BACKEND_URL ||
  (import.meta.env.DEV ? DEFAULT_DEV_BACKEND_URL : "")
).replace(/\/$/, "");

export function getBackendUrl(): string | null {
  return BACKEND_URL || null;
}

export function backendUrl(path: string): string {
  const baseUrl = getBackendUrl();

  if (!baseUrl) {
    throw new Error("Backend URL is not configured.");
  }

  return `${baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}

function readMessageFromJson(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const data = payload as Record<string, unknown>;

  for (const key of ["error", "message", "answer"]) {
    const value = data[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }

  return null;
}

export async function readApiError(
  response: Response,
  maxLength = 500,
): Promise<string> {
  const contentType = response.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    const data = await response.json().catch(() => undefined);
    const message = readMessageFromJson(data);

    if (message) {
      return message.slice(0, maxLength);
    }
  }

  const text = await response.text().catch(() => "");

  return (text || `Request failed with status ${response.status}`)
    .trim()
    .slice(0, maxLength);
}
