MediCore Backend API

1. API Overview

The backend is a Flask HTTP API.

Default local backend:

http://127.0.0.1:1819

The backend exposes:

health endpoints

authentication/session endpoints

email authentication endpoints

chat generation

conversation management

conversation message retrieval

CORS preflight routes

There is currently no versioned /api/v1 namespace.

Not currently implemented: formal API versioning.

2. Authentication

Authenticated requests use:

Authorization: Bearer <supabase-access-token>

Guest requests use:

X-Guest-Session-Id: <guest-session-id>

Some endpoints also accept:

guest_session_id

as a query or request-body parameter.

Authenticated protected endpoints require a verified email.

3. GET /

Returns a small HTML landing page confirming that the backend is running.

Response

200 OK
Content-Type: text/html

The page contains:

API status

frontend URL

/health

/get

This endpoint is intended primarily for human/service checks.

4. GET /health

Returns backend health status.

Request

GET /health

Response

{
  "status": "ok"
}

Status codes

Status

Meaning

200

Backend is responding

5. GET /health?deep=1

Performs a database connectivity check.

Request

GET /health?deep=1

Successful response

{
  "status": "ok",
  "database": "ok"
}

Failure response

{
  "status": "error",
  "database": "error"
}

Status:

503 Service Unavailable

The current deep health check verifies database connectivity only.

Not currently implemented: Pinecone/Groq/Supabase dependency health checks in /health?deep=1.

6. GET /api/auth/me

Returns the currently authenticated application user.

Authentication

Required:

Authorization: Bearer <token>

Email verification is required.

Request

GET /api/auth/me
Authorization: Bearer <supabase-access-token>

Response

{
  "user": {
    "id": "...",
    "auth_provider": "supabase",
    "external_id": "...",
    "email": "user@example.com",
    "display_name": "User",
    "avatar_url": null,
    "is_guest": false,
    "email_verified": true,
    "email_verified_at": "...",
    "last_login_at": "...",
    "created_at": "...",
    "updated_at": "..."
  }
}

Status codes

Status

Meaning

200

Authenticated user returned

401

Authentication missing/invalid/email unverified

429

Rate limit exceeded

503

Database failure

Rate limit:

120 requests / 60 seconds

7. POST /api/auth/session

Synchronizes the authenticated Supabase user with the application's local user/session database.

Request

POST /api/auth/session
Authorization: Bearer <supabase-access-token>

No request body is required.

Response

Same user representation as /api/auth/me.

{
  "user": {
    "id": "...",
    "auth_provider": "supabase",
    "external_id": "...",
    "email": "user@example.com",
    "display_name": "User",
    "avatar_url": null,
    "is_guest": false,
    "email_verified": true,
    "email_verified_at": "...",
    "last_login_at": "...",
    "created_at": "...",
    "updated_at": "..."
  }
}

Status codes

Status

Meaning

200

Session synchronized

401

Authentication failure

429

Rate limit exceeded

503

Database failure

Rate limit:

30 requests / 60 seconds

8. POST /api/auth/logout

Revokes the current backend-tracked authentication session.

Request

POST /api/auth/logout
Authorization: Bearer <supabase-access-token>

Response

{
  "ok": true
}

Status codes

Status

Meaning

200

Session revoked

401

Missing/invalid authentication

429

Rate limit exceeded

503

Database failure

Rate limit:

30 requests / 60 seconds

9. GET /api/auth/email-settings

Returns configured email authentication capabilities.

Request

GET /api/auth/email-settings

Response

{
  "email_provider_enabled": true,
  "signup_disabled": false,
  "mailer_autoconfirm": false
}

The exact values depend on Supabase configuration.

Status codes

Status

Meaning

200

Configuration returned

429

Rate limit exceeded

503

Supabase configuration/delivery failure

Rate limit:

60 requests / 60 seconds

10. POST /api/auth/send-otp

Requests a Supabase email OTP.

Request

POST /api/auth/send-otp
Content-Type: application/json

{
  "email": "user@example.com",
  "redirect_to": "https://example.com/auth/callback",
  "create_user": true
}

Accepted aliases include:

redirectTo
createUser

Validation

email must have a valid basic email format

redirect_to is required

redirect URL must use HTTP or HTTPS

redirect origin must be allowlisted

create_user must be boolean

Success

{
  "ok": true,
  "message": "Verification email request accepted by Supabase."
}

Status codes

Status

Meaning

200

Request accepted

400

Invalid request or redirect origin

429

Rate limit exceeded

502

Supabase delivery failure

503

Supabase configuration failure

Rate limit:

5 requests / 900 seconds

11. POST /api/auth/send-recovery

Requests a Supabase password recovery email.

Request

POST /api/auth/send-recovery
Content-Type: application/json

{
  "email": "user@example.com",
  "redirect_to": "https://example.com/auth/reset"
}

Success

{
  "ok": true,
  "message": "Password recovery email request accepted by Supabase."
}

Status codes

Status

Meaning

200

Request accepted

400

Invalid request/redirect

429

Rate limit exceeded

502

Supabase delivery failure

503

Supabase configuration failure

Rate limit:

5 requests / 900 seconds

12. POST /get

This is the primary AI chat endpoint.

Despite the legacy /get name, the endpoint is implemented as POST /get.

Request formats

The backend accepts JSON and form-style input.

Preferred JSON format:

POST /get
Content-Type: application/json

{
  "message": "What are common symptoms of iron deficiency?",
  "conversation_id": "conversation-id",
  "client_message_id": "client-message-id",
  "guest_session_id": "guest-session-id"
}

Accepted message field aliases:

msg
message
input

Accepted conversation ID aliases:

conversation_id
conversationId

Accepted client message ID aliases:

client_message_id
clientMessageId

Accepted guest session aliases:

guest_session_id
guestSessionId

A guest ID may also be supplied using:

X-Guest-Session-Id: <id>

Message limit

Maximum question length:

2000 characters

Successful response

The current API returns the answer as plain text.

Example:

HTTP/1.1 200 OK
Content-Type: text/plain; charset=utf-8

Iron deficiency can cause symptoms such as fatigue...

Error responses

Empty message:

400 Bad Request

Please enter a question.

Message over 2,000 characters:

413 Payload Too Large

Question is too long. Limit is 2000 characters.

Missing AI configuration:

503 Service Unavailable

The assistant is not configured. Please check server environment variables.

Generation failure:

500 Internal Server Error

Sorry, I could not generate a response right now. Please try again.

Important behavior

The generated answer is persisted after generation.

A persistence failure does not cause an otherwise successful AI response to fail. The failure is logged by the backend.

Rate limiting

Not currently implemented: endpoint-specific rate limiting for /get.

This is a significant production concern because /get invokes paid/external AI and vector services.

13. GET /api/conversations

Returns conversations belonging to the current user or guest session.

Authenticated request

GET /api/conversations
Authorization: Bearer <token>

Guest request

GET /api/conversations?guest_session_id=<guest-session-id>

Response

[
  {
    "id": "...",
    "external_id": "...",
    "title": "What are symptoms of...",
    "summary": null,
    "created_at": "...",
    "updated_at": "..."
  }
]

Status codes

Status

Meaning

200

Conversations returned

401

Authentication required/invalid

503

Database failure

14. POST /api/conversations

Creates a conversation.

Authenticated request

POST /api/conversations
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "Health consultation"
}

Guest request

{
  "title": "Guest consultation",
  "guest_session_id": "guest-session-id"
}

The guest ID may also be supplied through:

X-Guest-Session-Id

Response

201 Created

{
  "id": "...",
  "external_id": "...",
  "title": "Health consultation",
  "summary": null,
  "created_at": "...",
  "updated_at": "..."
}

Status codes

Status

Meaning

201

Created

400

Validation error

401

Authentication failure

503

Database failure

15. GET /api/conversations/<conversation_id>/messages

Returns all messages for a conversation accessible by the current identity.

Authenticated

GET /api/conversations/<conversation_id>/messages
Authorization: Bearer <token>

Guest

GET /api/conversations/<conversation_id>/messages?guest_session_id=<id>

Response

[
  {
    "id": "...",
    "conversation_id": "...",
    "role": "user",
    "content": "What is...",
    "parts": [
      {
        "type": "text",
        "text": "What is..."
      }
    ],
    "client_message_id": "...",
    "created_at": "..."
  },
  {
    "id": "...",
    "conversation_id": "...",
    "role": "assistant",
    "content": "....",
    "parts": [
      {
        "type": "text",
        "text": "..."
      }
    ],
    "client_message_id": null,
    "created_at": "..."
  }
]

Status codes

Status

Meaning

200

Messages returned

401

Authentication failure

404

Conversation not found

503

Database failure

16. PATCH /api/conversations/<conversation_id>

Renames a conversation.

Request

PATCH /api/conversations/<conversation_id>
Content-Type: application/json
Authorization: Bearer <token>

{
  "title": "Updated consultation title"
}

Guest sessions can additionally supply:

{
  "title": "Updated title",
  "guest_session_id": "guest-session-id"
}

Response

{
  "id": "...",
  "external_id": "...",
  "title": "Updated consultation title",
  "summary": null,
  "created_at": "...",
  "updated_at": "..."
}

Status codes

Status

Meaning

200

Updated

400

Missing/invalid title

401

Authentication failure

404

Conversation not found

503

Database failure

Maximum title length:

120 characters

17. DELETE /api/conversations/<conversation_id>

Deletes a conversation and its associated messages.

Request

DELETE /api/conversations/<conversation_id>
Authorization: Bearer <token>

Guest sessions can use:

DELETE /api/conversations/<conversation_id>?guest_session_id=<id>

Response

{
  "ok": true
}

Status codes

Status

Meaning

200

Deleted

401

Authentication failure

404

Conversation not found

503

Database failure

Messages are associated with the conversation using a cascading delete relationship.

18. CORS Preflight

The backend supports OPTIONS requests for:

/get
/api/auth/*
/api/conversations
/api/conversations/*

Allowed origins are configured using:

CORS_ALLOWED_ORIGINS
FRONTEND_URL

Development defaults also include common localhost ports.

19. Error Model

The API is not currently fully standardized.

Some endpoints return JSON:

{
  "error": "..."
}

while /get currently returns plain text errors.

Not currently implemented: a single versioned JSON error contract.

Recommended future format:

{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Question is too long.",
    "request_id": "..."
  }
}

20. API Features Not Currently Implemented

The following should not be assumed to exist:

/api/v1/... versioning

OpenAPI/Swagger specification

generated API client

streaming chat responses

source citations in chat responses

explicit /metrics

request IDs

distributed rate limiting

API keys for external consumers

webhook API

conversation-history-aware generation

automated API contract testing

These should be added only when the implementation actually supports them.
