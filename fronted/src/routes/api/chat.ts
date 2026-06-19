import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const backendUrl =
          process.env.BACKEND_URL || process.env.VITE_BACKEND_URL;

        if (!backendUrl) {
          return new Response(
            JSON.stringify({
              error: "Missing BACKEND_URL",
            }),
            {
              status: 500,
              headers: {
                "Content-Type": "application/json",
              },
            },
          );
        }

        try {
          const authHeader = request.headers.get("authorization") ?? "";
          const guestSessionId =
            request.headers.get("x-guest-session-id") ?? "";

          const body = await request.json().catch(() => undefined);

          if (!body || typeof body !== "object") {
            return new Response(
              JSON.stringify({
                error: "JSON body must be an object",
              }),
              {
                status: 400,
                headers: {
                  "Content-Type": "application/json",
                },
              },
            );
          }

          const payload = body as Record<string, unknown>;

          const response = await fetch(`${backendUrl.replace(/\/$/, "")}/get`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(authHeader
                ? {
                    Authorization: authHeader,
                  }
                : {}),
              ...(guestSessionId
                ? {
                    "X-Guest-Session-Id": guestSessionId,
                  }
                : {}),
            },
            body: JSON.stringify({
              message: payload.message ?? payload.input ?? payload.prompt ?? "",
              conversation_id:
                payload.conversation_id ?? payload.conversationId,
              client_message_id:
                payload.client_message_id ?? payload.clientMessageId,
              guest_session_id:
                payload.guest_session_id ??
                payload.guestSessionId ??
                guestSessionId,
            }),
          });

          const text = await response.text();
          const contentType =
            response.headers.get("content-type") ?? "text/plain; charset=utf-8";

          return new Response(text, {
            status: response.status,
            headers: {
              "Content-Type": contentType,
            },
          });
        } catch (error) {
          return new Response(
            JSON.stringify({
              error:
                error instanceof Error
                  ? error.message
                  : "Internal server error",
            }),
            {
              status: 500,
              headers: {
                "Content-Type": "application/json",
              },
            },
          );
        }
      },
    },
  },
});
