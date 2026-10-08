# MediCore — AI Medical Assistant

MediCore is a full-stack AI medical assistant built around retrieval-augmented generation (RAG). The application combines a React/TanStack frontend, a Flask API, Supabase Authentication, PostgreSQL/SQLite persistence, Pinecone vector search, Hugging Face embeddings, and Groq-hosted LLM inference.

The system is designed to provide general medical and health information grounded in an indexed medical knowledge base while maintaining conversation history and authenticated user sessions.

> **Medical safety notice:** MediCore is an educational and informational assistant. It is not a doctor, diagnostic system, emergency service, or substitute for professional medical care. AI-generated responses may be incomplete or incorrect. Users should verify important medical information with a qualified healthcare professional.

---

## Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Request Flow](#request-flow)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Local Development](#local-development)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Database Setup](#database-setup)
- [Authentication](#authentication)
- [Knowledge Base and RAG Index](#knowledge-base-and-rag-index)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Linting and Formatting](#linting-and-formatting)
- [Docker](#docker)
- [Deployment](#deployment)
- [Production Configuration](#production-configuration)
- [Security](#security)
- [Observability and Operations](#observability-and-operations)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Recommended Production Improvements](#recommended-production-improvements)
- [Development Workflow](#development-workflow)
- [License](#license)

---

## Overview

MediCore is split into two application layers:

```text
Browser
  │
  ▼
React / TanStack Start frontend
  │
  ├── Supabase Auth
  │
  └── Flask backend
        │
        ├── Request validation
        ├── Authentication / authorization
        ├── Conversation persistence
        ├── RAG pipeline
        │     ├── Hugging Face embeddings
        │     ├── Pinecone retrieval
        │     └── Groq LLM generation
        │
        └── PostgreSQL / SQLite
```

The backend owns the core application logic and persistence.

The frontend provides the user-facing application, authentication screens, chat interface, conversation history, and API integration.

---

## Key Features

### AI / RAG

- Retrieval-augmented medical question answering
- PDF-based medical knowledge ingestion
- Recursive document chunking
- Hugging Face `all-MiniLM-L6-v2` embeddings
- Pinecone vector storage and similarity retrieval
- Groq-hosted LLM generation
- Configurable retrieval count and generation parameters
- Safety-oriented medical system prompt

### Application

- Guest chat sessions
- Authenticated chat sessions
- Persistent conversations
- Conversation renaming
- Conversation deletion
- Message history
- Client message IDs
- Request size validation
- API rate limiting
- Health checks
- Database connectivity checks
- Structured database migrations

### Authentication

- Supabase email-based authentication
- Email verification
- Password sign-in
- Password reset flow
- Backend bearer-token validation
- Session persistence and revocation tracking
- Guest-session support

### Frontend

- React 19
- TanStack Router / TanStack Start
- Tailwind CSS
- Responsive chat UI
- Dark/light mode support
- Markdown rendering
- Conversation sidebar
- Loading and error states
- Client-side request cancellation
- Configurable backend URL

---

# Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 |
| Framework | TanStack Start |
| Routing | TanStack Router |
| Styling | Tailwind CSS |
| Backend | Flask 3 |
| Python | 3.10 |
| ORM | SQLAlchemy 2 |
| Database migrations | Alembic |
| Production DB | PostgreSQL |
| Development DB | SQLite |
| Authentication | Supabase Auth |
| Vector database | Pinecone |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | Groq |
| RAG orchestration | LangChain |
| PDF ingestion | PyPDF |
| Backend server | Gunicorn |
| Frontend package manager | npm / Bun artifacts are present |
| Containerization | Docker |
| Deployment configuration | Render |

---

# System Architecture

## High-Level Architecture

```text
                           ┌───────────────────────┐
                           │       Browser         │
                           │ React / TanStack Start│
                           └───────────┬───────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
              ┌───────────────┐                 ┌───────────────┐
              │ Supabase Auth │                 │ Flask Backend │
              └───────────────┘                 └───────┬───────┘
                                                        │
                         ┌──────────────────────────────┼──────────────────────┐
                         │                              │                      │
                         ▼                              ▼                      ▼
                  ┌─────────────┐               ┌──────────────┐       ┌──────────────┐
                  │ PostgreSQL  │               │   Pinecone   │       │    Groq LLM  │
                  │ / SQLite    │               │ Vector Index │       │   Inference  │
                  └─────────────┘               └──────────────┘       └──────────────┘
                                                       ▲
                                                       │
                                                ┌──────┴───────┐
                                                │ Hugging Face │
                                                │  Embeddings  │
                                                └──────────────┘
```

---

# RAG Pipeline

The backend builds the RAG chain lazily and caches it per process.

The retrieval pipeline is:

```text
User Question
      │
      ▼
Request validation
      │
      ▼
Hugging Face embedding
      │
      ▼
Pinecone similarity search
      │
      ▼
Top-K document chunks
      │
      ▼
LangChain retrieval chain
      │
      ▼
Groq LLM
      │
      ▼
Safety-oriented system prompt
      │
      ▼
Generated answer
      │
      ▼
Conversation persistence
```

The current default configuration uses:

```text
Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Retriever:
similarity search

Default K:
3

LLM:
openai/gpt-oss-120b

Temperature:
0.2

Maximum output tokens:
1024
```

---

# Knowledge Base

The repository currently contains a medical reference PDF under:

```text
backend/data/
└── Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf
```

The indexing script:

```text
backend/store_index.py
```

uses the following flow:

```text
PDF
 │
 ▼
PyPDFLoader
 │
 ▼
Document metadata cleanup
 │
 ▼
RecursiveCharacterTextSplitter
 │
 ▼
500 character chunks
 │
 ▼
50 character overlap
 │
 ▼
Sentence-transformer embeddings
 │
 ▼
Pinecone
```

The embedding implementation is cached so the same model is not unnecessarily initialized repeatedly inside a process.

---

# Request Flow

A typical chat request works as follows:

```text
Frontend
   │
   │ POST /get
   ▼
Flask
   │
   ├── Validate payload
   ├── Validate message length
   ├── Resolve user / guest identity
   │
   ▼
RAG chain
   │
   ├── Pinecone retrieval
   └── Groq generation
   │
   ▼
Generated answer
   │
   ├── Return to frontend
   │
   └── Persist:
       ├── user message
       ├── assistant response
       └── conversation metadata
```

---

# Repository Structure

```text
Medical-chat-bot-AI/
│
├── backend/
│   ├── .dockerignore
│   ├── .env.example
│   ├── .gitignore
│   ├── Dockerfile
│   ├── LICENSE
│   ├── README.md
│   ├── alembic.ini
│   ├── app.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── setup.py
│   ├── store_index.py
│   │
│   ├── data/
│   │   └── Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf
│   │
│   ├── migrations/
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   │       ├── 20260610_0001_initial_database.py
│   │       └── 20260610_0002_auth_user_metadata.py
│   │
│   ├── scripts/
│   │   └── seed.py
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── database.py
│   │   ├── helper.py
│   │   ├── models.py
│   │   ├── prompt.py
│   │   ├── rate_limit.py
│   │   ├── repositories.py
│   │   ├── schemas.py
│   │   └── supabase_email.py
│   │
│   └── tests/
│       └── test_app.py
│
├── fronted/
│   ├── .env.example
│   ├── .gitignore
│   ├── LICENSE
│   ├── README.md
│   ├── package.json
│   ├── package-lock.json
│   ├── bun.lock
│   ├── bunfig.toml
│   ├── render.yaml
│   ├── tsconfig.json
│   ├── vite.config.ts
│   │
│   ├── src/
│   │   ├── components/
│   │   ├── integrations/
│   │   │   └── supabase/
│   │   ├── lib/
│   │   ├── routes/
│   │   ├── server.ts
│   │   ├── start.ts
│   │   └── styles.css
│   │
│   └── supabase/
│       ├── config.toml
│       └── migration/
│
└── README.md
```

---

# Prerequisites

Install the following before starting development:

### Backend

- Python 3.10
- pip
- PostgreSQL for production-style local development
- Git

### Frontend

- Node.js 22.x recommended by the current Render configuration
- npm

### External services

You need credentials/configuration for:

- Supabase
- Pinecone
- Groq

---

# Environment Configuration

There are separate environment files for the backend and frontend.

## Backend

Copy:

```bash
cd backend
cp .env.example .env
```

On Windows:

```powershell
copy .env.example .env
```

Example:

```env
PINECONE_API_KEY=
GROQ_API_KEY=

DATABASE_URL=
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=1800
DATABASE_AUTO_CREATE=0

SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
SUPABASE_JWT_SECRET=
SUPABASE_JWT_AUDIENCE=authenticated
SUPABASE_JWT_ALGORITHMS=HS256

FRONTEND_URL=http://127.0.0.1:8080

CORS_ALLOWED_ORIGINS=http://127.0.0.1:8080,http://localhost:8080

AUTH_REDIRECT_ALLOWED_ORIGINS=http://127.0.0.1:8080,http://localhost:8080

PINECONE_INDEX_NAME=medical-chatbot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

RETRIEVER_K=3

GROQ_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0.2
GROQ_MAX_TOKENS=1024

FLASK_DEBUG=0
FLASK_RUN_HOST=0.0.0.0
PORT=1819

WEB_CONCURRENCY=1
GUNICORN_TIMEOUT=120
LOG_LEVEL=INFO

PDF_DATA_DIR=data
```

## Frontend

Copy:

```bash
cd fronted
cp .env.example .env
```

Example:

```env
VITE_SUPABASE_URL=
VITE_SUPABASE_PUBLISHABLE_KEY=
VITE_SUPABASE_PROJECT_ID=
VITE_BACKEND_URL=http://127.0.0.1:1819
```

For production:

```env
VITE_SUPABASE_URL=https://<project>.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=<publishable-key>
VITE_SUPABASE_PROJECT_ID=<project-id>
VITE_BACKEND_URL=https://<backend-domain>
```

---

# Backend Setup

From the repository root:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

---

# Frontend Setup

From the repository root:

```bash
cd fronted
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend is configured to use:

```text
http://127.0.0.1:1819
```

as the development backend unless overridden by `VITE_BACKEND_URL`.

---

# Database Setup

## Development

If `DATABASE_URL` is not defined, the backend falls back to:

```text
backend/instance/medicore.sqlite3
```

SQLite is intended for local development only.

## Production

Use PostgreSQL.

Example:

```env
DATABASE_URL=postgresql+psycopg://username:password@host:5432/medicore
```

Run migrations:

```bash
cd backend
alembic upgrade head
```

To generate a migration after changing SQLAlchemy models:

```bash
alembic revision --autogenerate -m "describe change"
```

Do not rely on `DATABASE_AUTO_CREATE=1` in production. Production schema management should be migration-driven.

---

# Seed Data

A small development seed script is available:

```bash
cd backend
python -m scripts.seed
```

It creates a demo guest conversation.

Seed data should not be used as a substitute for real application fixtures or production data initialization.

---

# Authentication

Authentication is handled primarily by Supabase Auth.

Supported flows include:

### Registration

```text
Sign Up
   │
   ▼
Email OTP
   │
   ▼
Verification
   │
   ▼
Password creation
   │
   ▼
Authenticated session
```

### Login

```text
Email + Password
      │
      ▼
Supabase Auth
      │
      ▼
Verified user
      │
      ▼
Backend session synchronization
```

### Password recovery

```text
Forgot Password
      │
      ▼
Supabase recovery email
      │
      ▼
Recovery session
      │
      ▼
New password
```

The browser stores and refreshes the Supabase session using the Supabase client.

The backend validates bearer tokens and records application-level user/session metadata.

---

# Guest Sessions

Users can interact with the chatbot without creating an authenticated account.

Guest identity is tracked using a browser-generated session ID.

The frontend stores:

```text
medicore_guest_session_id
```

in browser local storage.

The backend uses the guest session to associate:

- conversations
- messages
- conversation history

Guest sessions should be treated as convenience sessions rather than strong authentication.

---

# API Reference

## Health

### `GET /health`

Returns a lightweight health response.

```bash
curl http://127.0.0.1:1819/health
```

Response:

```json
{
  "status": "ok"
}
```

### `GET /health?deep=1`

Performs a database connectivity check.

```bash
curl "http://127.0.0.1:1819/health?deep=1"
```

---

## Chat

### `POST /get`

Accepts a user question and returns the generated answer.

Example:

```json
{
  "message": "What are common causes of fever?",
  "conversation_id": "guest",
  "guest_session_id": "example-session"
}
```

The endpoint supports both JSON and form-style input.

Maximum question length:

```text
2000 characters
```

Possible responses include:

- `200` — response generated
- `400` — invalid input
- `413` — message too long
- `500` — generation failure
- `503` — missing runtime configuration

---

# Authentication API

## `GET /api/auth/me`

Returns the currently authenticated backend user.

Requires:

```http
Authorization: Bearer <token>
```

---

## `POST /api/auth/session`

Synchronizes an authenticated Supabase session with the backend application database.

Requires:

```http
Authorization: Bearer <token>
```

---

## `POST /api/auth/logout`

Revokes the backend application's stored session token hash.

Requires:

```http
Authorization: Bearer <token>
```

---

## `GET /api/auth/email-settings`

Returns configured Supabase email authentication settings.

---

## `POST /api/auth/send-otp`

Requests a Supabase verification email.

Example:

```json
{
  "email": "user@example.com",
  "redirect_to": "https://app.example.com/verify-otp",
  "create_user": true
}
```

---

## `POST /api/auth/send-recovery`

Requests a password recovery email.

Example:

```json
{
  "email": "user@example.com",
  "redirect_to": "https://app.example.com/reset-password"
}
```

---

# Conversation API

## `GET /api/conversations`

Returns the user's saved conversations.

Authenticated requests use bearer authentication.

Guest requests use:

```text
guest_session_id
```

---

## `POST /api/conversations`

Creates a conversation.

Example:

```json
{
  "title": "Fever consultation"
}
```

---

## `GET /api/conversations/{conversation_id}/messages`

Returns messages for a specific conversation.

---

## `PATCH /api/conversations/{conversation_id}`

Renames a conversation.

Example:

```json
{
  "title": "Fever and hydration"
}
```

---

## `DELETE /api/conversations/{conversation_id}`

Deletes a conversation and its associated messages.

---

# Rate Limiting

The backend includes an in-process rate limiter.

Current protected scopes include authentication-related endpoints and authentication/session operations.

The limiter returns:

```http
429 Too Many Requests
```

with:

```http
Retry-After: <seconds>
```

### Important production limitation

The current limiter is process-local memory.

That means it is not suitable as the sole rate-limiting mechanism when the application runs across multiple workers or multiple instances.

A distributed limiter should eventually use infrastructure such as Redis or an API gateway.

---

# Security

The backend currently implements several baseline controls:

- bearer-token authentication
- token hashing before session persistence
- request validation
- request body size limits
- allowed auth redirect origins
- configurable CORS origins
- response security headers
- `X-Content-Type-Options`
- `X-Frame-Options`
- `Referrer-Policy`
- Content Security Policy
- rate limiting
- PostgreSQL connection pooling
- database migrations
- secrets supplied through environment variables

Do not commit:

```text
.env
API keys
JWT secrets
database passwords
Supabase service-role secrets
Pinecone API keys
Groq API keys
```

---

# RAG Index Management

To build the Pinecone index:

```bash
cd backend
python store_index.py
```

The script expects:

```text
backend/data/*.pdf
```

The default index configuration is:

```env
PINECONE_INDEX_NAME=medical-chatbot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

The embedding model produces 384-dimensional vectors.

If the embedding model changes, the Pinecone index dimension and stored vectors must be recreated accordingly.

---

# Testing

Backend tests use Pytest.

Run:

```bash
cd backend
pytest -q
```

The existing suite covers:

- health endpoint
- deep health check
- authentication input validation
- auth redirect validation
- chat validation
- maximum message length
- RAG invocation behavior
- guest persistence
- conversation persistence
- configuration failures
- environment validation

The RAG chain is mocked in tests where necessary so tests do not require live Groq or Pinecone calls.

---

# Static Validation

Run Python compilation checks:

```bash
cd backend
python -m compileall app.py store_index.py src
```

Run the test suite:

```bash
python -m pytest -q
```

---

# Frontend Checks

Run:

```bash
cd fronted
npm run lint
```

Build:

```bash
npm run build
```

Run the production server locally after building:

```bash
npm start
```

---

# Docker

The backend includes a Dockerfile.

Build:

```bash
cd backend
docker build -t medicore-backend .
```

Run:

```bash
docker run --env-file .env -p 1819:1819 medicore-backend
```

The image uses Gunicorn and binds to:

```text
0.0.0.0:${PORT}
```

---

# Deployment

The current repository contains Render deployment configuration for the frontend.

A practical production deployment consists of:

```text
Frontend
  │
  └── Render Web Service

Backend
  │
  └── Render / Docker Web Service

Authentication
  │
  └── Supabase

Relational database
  │
  └── PostgreSQL / Supabase PostgreSQL

Vector database
  │
  └── Pinecone

LLM provider
  │
  └── Groq
```

---

# Backend Deployment

A production backend should provide:

```env
PINECONE_API_KEY=...
GROQ_API_KEY=...

DATABASE_URL=postgresql+psycopg://...

SUPABASE_URL=...
SUPABASE_PUBLISHABLE_KEY=...
SUPABASE_JWT_SECRET=...

PINECONE_INDEX_NAME=medical-chatbot

RETRIEVER_K=3
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0.2
GROQ_MAX_TOKENS=1024

FRONTEND_URL=https://frontend.example.com
CORS_ALLOWED_ORIGINS=https://frontend.example.com
AUTH_REDIRECT_ALLOWED_ORIGINS=https://frontend.example.com

DATABASE_AUTO_CREATE=0

WEB_CONCURRENCY=1
GUNICORN_TIMEOUT=120
LOG_LEVEL=INFO
```

After deployment:

```bash
alembic upgrade head
```

Then verify:

```bash
curl https://backend.example.com/health
```

And:

```bash
curl "https://backend.example.com/health?deep=1"
```

---

# Frontend Deployment

The repository includes:

```text
fronted/render.yaml
```

The current deployment configuration builds using:

```bash
npm install && npm run build
```

and starts using:

```bash
npm start
```

The frontend must be configured with:

```env
VITE_SUPABASE_URL=...
VITE_SUPABASE_PUBLISHABLE_KEY=...
VITE_SUPABASE_PROJECT_ID=...
VITE_BACKEND_URL=https://backend.example.com
```

The backend must then allow the frontend domain through:

```env
CORS_ALLOWED_ORIGINS
AUTH_REDIRECT_ALLOWED_ORIGINS
FRONTEND_URL
```

---

# Production Configuration

Before production traffic is accepted, verify all of the following:

### Infrastructure

- PostgreSQL configured
- Pinecone index created
- Knowledge-base documents indexed
- Supabase Auth configured
- Groq API access configured
- HTTPS enabled
- Frontend and backend domains configured

### Application

- `FLASK_DEBUG=0`
- `DATABASE_AUTO_CREATE=0`
- production `DATABASE_URL`
- production CORS allowlist
- production auth redirect allowlist
- correct Supabase JWT configuration
- correct Pinecone index
- appropriate LLM configuration

### Operational

- health checks enabled
- application logs collected
- error monitoring configured
- dependency updates monitored
- backups configured
- database migrations tested
- secret rotation procedure defined

---

# Observability and Operations

The application currently provides basic logging through Python's logging framework.

Production monitoring should be expanded to include:

### Metrics

- request count
- request latency
- HTTP error rate
- RAG generation latency
- Pinecone retrieval latency
- database latency
- authentication failures
- rate-limit events
- token/LLM usage
- model failure rate

### Logging

Log structured events for:

```text
request_started
request_completed
authentication_failed
rag_started
rag_completed
rag_failed
database_error
rate_limit_triggered
```

Do not log:

- passwords
- raw JWTs
- API keys
- raw authentication tokens
- sensitive medical data unless there is a documented and justified reason

---

# Troubleshooting

## Backend starts but `/get` returns 503

Check:

```env
PINECONE_API_KEY
GROQ_API_KEY
PINECONE_INDEX_NAME
```

Also verify that the configured Pinecone index exists and has vectors.

---

## Backend starts but authentication fails

Check:

```env
SUPABASE_URL
SUPABASE_PUBLISHABLE_KEY
SUPABASE_JWT_SECRET
SUPABASE_JWT_AUDIENCE
```

Verify the frontend and backend use the same Supabase project.

---

## CORS errors in the browser

Make sure the frontend origin is included in:

```env
CORS_ALLOWED_ORIGINS
```

and:

```env
FRONTEND_URL
```

For authentication redirects, also configure:

```env
AUTH_REDIRECT_ALLOWED_ORIGINS
```

---

## Chat works locally but not in production

Check:

```text
1. Backend URL
2. Pinecone index
3. Groq API key
4. Supabase configuration
5. CORS
6. Database migrations
7. Production port
8. Worker startup command
```

---

## RAG returns poor answers

Inspect:

```text
- source document quality
- chunk size
- chunk overlap
- retriever K
- embedding model
- Pinecone index contents
- prompt quality
- model selection
```

The current chunking configuration is:

```text
chunk_size = 500
chunk_overlap = 50
```

This is a reasonable baseline, not a validated medical retrieval configuration.

---

## Pinecone dimension mismatch

The current embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and produces 384-dimensional vectors.

Do not reuse an index built with a different embedding dimension.

Rebuild the index after changing embedding models.

---

# Known Limitations

This repository should not currently be described as a fully production-hardened healthcare platform.

Important limitations include:

### 1. Single-process in-memory rate limiter

The current rate limiter does not provide distributed enforcement.

### 2. Limited test coverage

There is a meaningful backend test suite, but there is no comparable frontend test suite covering the complete UI and authentication flows.

### 3. No visible repository CI pipeline

There is currently no `.github/workflows` CI setup in the repository.

### 4. No centralized API specification

The API exists in Flask route implementations, but there is no OpenAPI/Swagger contract generated from the backend.

### 5. Retrieval quality is not formally evaluated

The repository does not currently include:

- golden questions
- retrieval precision/recall evaluation
- answer groundedness evaluation
- hallucination benchmarks
- regression evaluation for medical responses

### 6. Limited observability

The project has logging and health checks but does not include a complete production observability stack.

### 7. Frontend dependency/tooling duplication

Both:

```text
package-lock.json
bun.lock
```

are present.

The project should standardize on one package manager unless there is an intentional reason to maintain both.

### 8. Large source document in Git

The medical PDF is stored directly in the repository.

For a growing production knowledge base, documents should generally be managed through dedicated storage and a repeatable ingestion pipeline rather than treating the repository as the document warehouse.

### 9. Medical safety requires additional controls

A system prompt and disclaimer are not enough by themselves for a high-risk medical production system.

Safety evaluation, escalation policies, source provenance, monitoring, and clinical review would be required for a substantially higher assurance level.

---

# Recommended Production Improvements

These are the highest-value changes I would make before presenting this as a mature production project.

## P0 — Do before production

### Add CI

Create:

```text
.github/workflows/ci.yml
```

CI should run at minimum:

```text
Python:
- compile check
- pytest
- dependency checks

Frontend:
- npm ci
- lint
- build
```

### Add security scanning

Enable:

- Dependabot
- secret scanning
- push protection
- code scanning

GitHub specifically recommends these repository security controls for public projects.

### Add `SECURITY.md`

Define:

- supported versions
- vulnerability reporting process
- response expectations
- disclosure policy

GitHub recommends a repository security policy for vulnerability reporting.

### Add root-level project documentation

The repository currently has backend and frontend README files but lacks a top-level README that documents the actual product architecture.

---

# P1 — Strongly recommended

Add:

```text
docs/
├── architecture.md
├── api.md
├── deployment.md
├── development.md
├── rag-pipeline.md
└── security.md
```

Add an OpenAPI specification:

```text
docs/openapi.yaml
```

Add frontend testing with:

```text
Vitest
Testing Library
Playwright
```

Add backend integration tests covering:

- Supabase authentication
- conversation ownership
- unauthorized access
- deleted conversations
- database migration compatibility
- Pinecone/RAG integration

---

# P2 — Production maturity

Add:

```text
.github/
├── workflows/
├── ISSUE_TEMPLATE/
└── pull_request_template.md
```

Also consider:

```text
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
SUPPORT.md
CHANGELOG.md
```

GitHub explicitly recommends README, contribution guidance, code of conduct, license and security documentation as part of maintaining a healthy repository.

---

# Recommended Future Architecture

For a larger production deployment:

```text
                         ┌──────────────────┐
                         │       CDN        │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │    Frontend      │
                         │ React / TanStack  │
                         └────────┬─────────┘
                                  │
                           HTTPS / API
                                  │
                    ┌─────────────▼─────────────┐
                    │       API Gateway        │
                    └─────────────┬─────────────┘
                                  │
                  ┌───────────────▼───────────────┐
                  │        Flask API              │
                  │ auth / validation / chat      │
                  └──────┬───────────────┬────────┘
                         │               │
                 ┌───────▼──────┐  ┌────▼───────┐
                 │ Redis        │  │ PostgreSQL │
                 │ rate limits  │  │ app data   │
                 └──────────────┘  └────────────┘
                         │
                  ┌──────▼─────────┐
                  │ RAG service    │
                  └──────┬─────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
       ┌──────▼──────┐        ┌─────▼─────┐
       │  Pinecone   │        │  Groq LLM │
       └─────────────┘        └───────────┘
```

At scale, Redis should replace the current process-local rate-limit store, and API/application services should have explicit metrics and distributed tracing.

---

# Development Workflow

Recommended local workflow:

```bash
git checkout -b feature/<name>

# Backend
cd backend
python -m pytest -q

# Frontend
cd ../fronted
npm run lint
npm run build
```

Commit only after local validation succeeds.

For database changes:

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

Do not manually edit an already-applied production migration. Add a new migration instead.

---

# Production Readiness Checklist

Use this checklist before the first real deployment:

```text
[ ] Root README updated
[ ] CI pipeline added
[ ] Backend tests passing
[ ] Frontend tests added
[ ] Frontend lint passing
[ ] Production build passing
[ ] PostgreSQL configured
[ ] Alembic migrations verified
[ ] Pinecone index created
[ ] Knowledge base indexed
[ ] Groq credentials configured
[ ] Supabase Auth configured
[ ] JWT verification configured
[ ] CORS restricted to real origins
[ ] Auth redirect allowlist configured
[ ] HTTPS enabled
[ ] Distributed rate limiting configured
[ ] Secret scanning enabled
[ ] Dependabot enabled
[ ] Code scanning enabled
[ ] SECURITY.md added
[ ] Monitoring configured
[ ] Error tracking configured
[ ] Database backups configured
[ ] Recovery procedure documented
[ ] Medical safety evaluation performed
```

---

# Project Status

The repository currently contains the core components required for an end-to-end working product:

- full-stack frontend
- Flask backend
- RAG pipeline
- Pinecone vector retrieval
- Groq LLM integration
- Supabase authentication
- persistent conversation storage
- database migrations
- guest sessions
- Docker support
- deployment configuration
- backend automated tests

The next stage is not adding more UI features. The highest-value work is improving the engineering foundation around the existing product:

```text
CI/CD
+
testing
+
security
+
observability
+
API contracts
+
distributed infrastructure
+
RAG evaluation
```

---

# License

The repository currently contains separate license files under:

```text
backend/LICENSE
fronted/LICENSE
```

Before publishing the project as a single distributable product, the repository should define and document one clear top-level licensing policy covering the complete codebase and separately identify any third-party or externally licensed medical reference material.

---

# Disclaimer

MediCore provides AI-generated health information for educational purposes.

It should not be used for:

- emergency diagnosis
- emergency treatment
- medication prescribing
- personalized treatment decisions
- replacing a doctor or other licensed healthcare professional

For severe or rapidly worsening symptoms, users should seek appropriate professional medical care.
