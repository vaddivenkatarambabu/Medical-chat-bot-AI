MediCore Medical Chatbot — System Architecture

1. Purpose

MediCore is a full-stack medical/health information chatbot built around a Retrieval-Augmented Generation (RAG) architecture.

The system combines:

A React/TanStack frontend

A Flask Python backend

Supabase Authentication

SQLAlchemy-based persistence

SQLite for local development

PostgreSQL for production deployments

Pinecone as the vector database

Hugging Face all-MiniLM-L6-v2 embeddings

LangChain for RAG orchestration

Groq for LLM inference

PDF-based medical knowledge ingestion

Docker support for the backend

Render deployment configuration for the frontend

The system is designed primarily for general health education and symptom guidance. It is not designed to replace professional medical diagnosis or treatment.

2. High-Level Architecture

                           ┌─────────────────────────┐
                           │        User             │
                           │ Browser / Web Client    │
                           └────────────┬────────────┘
                                        │
                                        ▼
                    ┌──────────────────────────────────┐
                    │ React + TanStack Start Frontend  │
                    │                                  │
                    │ - React 19                       │
                    │ - TanStack Router/Start          │
                    │ - TypeScript                     │
                    │ - Vite                           │
                    │ - Tailwind CSS                   │
                    │ - Supabase JS                    │
                    └───────────────┬──────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
          ┌────────────────────┐        ┌─────────────────────┐
          │ Supabase Auth      │        │ TanStack API Route  │
          │                    │        │ /api/chat            │
          │ Email / OTP /      │        │                     │
          │ session management │        │ Proxy to backend    │
          └─────────┬──────────┘        └──────────┬──────────┘
                    │                              │
                    │ Bearer token                 │ HTTP
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌────────────────────────────────┐
                    │ Flask Backend                   │
                    │                                │
                    │ Authentication                 │
                    │ Validation                     │
                    │ Conversation API               │
                    │ Chat API                       │
                    │ CORS / Security Headers        │
                    │ Persistence                    │
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────┼──────────────────┐
                    │              │                  │
                    ▼              ▼                  ▼
             ┌────────────┐ ┌─────────────┐ ┌────────────────┐
             │ PostgreSQL │ │ Supabase    │ │ RAG Pipeline   │
             │ / SQLite   │ │ Auth API    │ │                │
             └────────────┘ └─────────────┘ └───────┬────────┘
                                                    │
                                      ┌─────────────┼─────────────┐
                                      │             │             │
                                      ▼             ▼             ▼
                                  Embeddings    Pinecone       Groq
                                  Hugging Face  Vector DB      LLM

3. Repository Structure

The current repository is organized into backend, frontend, and repository-level documentation/configuration.

Medical-chat-bot-AI/
├── backend/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── pyproject.toml
│   ├── alembic.ini
│   ├── .env.example
│   ├── store_index.py
│   ├── setup.py
│   ├── data/
│   │   └── Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf
│   ├── migrations/
│   ├── scripts/
│   │   └── seed.py
│   ├── src/
│   │   ├── auth.py
│   │   ├── database.py
│   │   ├── helper.py
│   │   ├── models.py
│   │   ├── prompt.py
│   │   ├── rate_limit.py
│   │   ├── repositories.py
│   │   ├── schemas.py
│   │   └── supabase_email.py
│   └── tests/
│       └── test_app.py
│
├── fronted/
│   ├── package.json
│   ├── package-lock.json
│   ├── bun.lock
│   ├── render.yaml
│   ├── vite.config.ts
│   ├── .env.example
│   └── src/
│       ├── lib/
│       ├── routes/
│       └── ...
│
├── docs/
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── SUPPORT.md
├── CODE_OF_CONDUCT.md
└── CHANGELOG.md

fronted/ is intentionally documented using the repository's current directory name. Renaming it to frontend/ would be a separate repository-wide change and is not assumed here.

4. Frontend Architecture

The frontend is a TypeScript application based on React 19 and TanStack Start.

Main technologies

Layer

Technology

UI

React 19

Application framework

TanStack Start

Routing

TanStack Router

Build

Vite

Language

TypeScript

Styling

Tailwind CSS

Authentication client

Supabase JS

HTTP communication

Fetch API

Server-side chat proxy

TanStack Start route

Production server

srvx

The frontend is responsible for:

Rendering the chat interface.

Managing the authenticated Supabase session.

Generating and storing guest session IDs.

Managing conversation lists.

Loading conversation messages.

Sending chat requests.

Forwarding authentication and guest-session information to the backend.

5. Backend Architecture

The backend is a Flask application.

The main application entry point is:

backend/app.py

The application is responsible for:

HTTP routing

Request validation

Authentication

Authorization

Conversation persistence

Chat generation

RAG initialization

CORS

Security headers

Health checks

Email authentication integration

The RAG chain is initialized lazily through:

@lru_cache(maxsize=1)
def get_rag_chain():
    ...

This means the RAG chain is constructed once per Python process and then reused.

This reduces repeated initialization of:

Hugging Face embeddings

Pinecone vector store

Retriever

Groq LLM

LangChain chains

If multiple backend worker processes are used, each process has its own cached RAG chain.

6. Authentication Architecture

MediCore supports two identity modes.

Authenticated users

Authenticated users use Supabase Auth.

The frontend obtains a Supabase access token and sends:

Authorization: Bearer <access-token>

The backend validates the token using the configured Supabase authentication mechanism.

Authenticated users must have a verified email for protected authenticated operations.

The backend maps the external Supabase identity into the local app_users table.

Guest users

Guest users do not authenticate through Supabase.

The frontend creates a random identifier using nanoid() and stores it locally:

medicore_guest_session_id

The identifier is sent using:

X-Guest-Session-Id: <guest-session-id>

or through the request body/query parameters where supported.

The backend creates a guest user record using this identifier.

Security implication

A guest session ID is an identifier, not strong authentication.

Anyone possessing the identifier may potentially act as that guest.

Therefore:

Not currently implemented: cryptographically authenticated guest accounts.

For higher-security production use, guest sessions should eventually be upgraded to server-issued, signed or otherwise protected session credentials.

7. Database Architecture

The persistence layer uses SQLAlchemy.

Local development

The default database is:

SQLite

Default path:

instance/medicore.sqlite3

Production

PostgreSQL is supported through:

psycopg

The application automatically normalizes common PostgreSQL URL formats.

Production database connections support:

connection pooling

pool pre-ping

pool timeout

pool recycle

configurable pool size

configurable overflow

8. Current Database Entities

The main application entities are:

app_users

Stores application-level user records.

Supports:

Supabase users

guest users

user_sessions

Stores hashed authentication-token session metadata.

The raw access token is not stored in the database.

chat_conversations

Stores conversation metadata.

Important fields include:

id

user_id

external_id

guest_session_id

title

summary

source

timestamps

chat_messages

Stores individual conversation messages.

Important fields include:

conversation_id

user_id

role

content

parts

client_message_id

created_at

9. Important Conversation Architecture Limitation

Conversation history is persisted in the database.

However, the current /get implementation does not retrieve previous conversation messages and inject them into the RAG/LLM prompt.

Current generation flow:

Current user question
        │
        ▼
Pinecone retrieval
        │
        ▼
Retrieved context
        │
        ▼
System prompt + current question
        │
        ▼
Groq

It is not currently:

Previous conversation
        +
Current question
        +
Retrieved context
        │
        ▼
LLM

Therefore, persistence provides conversation history to the UI but does not currently provide conversational memory to the model.

Not currently implemented: LLM-aware conversational memory.

A future implementation should add controlled conversation-history retrieval with:

token limits

truncation

summarization

prompt-injection isolation

per-user authorization checks

10. RAG Architecture

The project uses Retrieval-Augmented Generation.

Ingestion

PDF files
   │
   ▼
PyPDFLoader
   │
   ▼
Minimal metadata
   │
   ▼
RecursiveCharacterTextSplitter
   │
   ▼
500-character chunks
50-character overlap
   │
   ▼
Hugging Face embeddings
   │
   ▼
384-dimensional vectors
   │
   ▼
Pinecone

Runtime

User question
     │
     ▼
Flask /get
     │
     ▼
Embedding model
     │
     ▼
Pinecone similarity search
     │
     ▼
Top K documents
     │
     ▼
LangChain retrieval chain
     │
     ▼
System prompt + retrieved context + question
     │
     ▼
Groq LLM
     │
     ▼
Answer

The default retrieval count is:

RETRIEVER_K=3

11. Vector Database

Pinecone is the vector database.

The indexing script creates a serverless index with:

dimension = 384
metric = cosine

Default configuration:

PINECONE_INDEX_NAME=medical-chatbot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

The vector database is external to the Flask application.

The backend connects to an existing Pinecone index at runtime.

12. Embedding Model

The project uses:

sentence-transformers/all-MiniLM-L6-v2

through:

langchain_huggingface.HuggingFaceEmbeddings

Embeddings are normalized:

encode_kwargs={"normalize_embeddings": True}

The resulting vector size is:

384

The same embedding configuration must be used for both indexing and retrieval.

Changing the embedding model requires rebuilding/reindexing the Pinecone index.

13. LLM Layer

The project uses Groq through LangChain.

Default model:

openai/gpt-oss-120b

Configurable parameters include:

GROQ_MODEL
GROQ_TEMPERATURE
GROQ_MAX_TOKENS

Current defaults:

temperature = 0.2
max_tokens = 1024

The low temperature is appropriate for a system where predictable and grounded responses are preferred over creative generation.

14. Prompt Layer

The system prompt is defined in:

backend/src/prompt.py

It instructs the model to:

provide general health education

avoid diagnosis certainty

avoid personalized medication dosing

avoid fabricated facts

identify emergency symptoms

use retrieved context as the main medical source

state uncertainty where context is insufficient

answer non-medical questions normally

provide concise, structured responses

This prompt is part of the application's safety boundary but should not be treated as a complete medical safety control.

15. External Services

The current architecture depends on:

Service

Purpose

Supabase Auth

User authentication and email authentication

Pinecone

Vector storage and similarity retrieval

Groq

LLM inference

Hugging Face/Sentence Transformers

Embedding generation

PostgreSQL provider

Production relational persistence

Render

Frontend deployment configuration

16. Deployment Architecture

The frontend has a Render Web Service configuration:

fronted/render.yaml

The frontend is built and served as a Node/TanStack Start application.

The backend has a Dockerfile:

backend/Dockerfile

The Docker image uses:

python:3.10-slim-bookworm

and starts Gunicorn.

The backend can therefore be deployed to a container-compatible platform.

Not currently implemented: a repository-level infrastructure definition for the backend deployment equivalent to the frontend render.yaml.

For reproducible production infrastructure, add either:

a backend Render service definition,

Terraform,

another infrastructure-as-code solution,

or documented platform configuration checked into the repository.

17. Runtime Data Flow

Authenticated chat

Browser
  │
  ├── Supabase session
  │
  ▼
Frontend
  │
  │ Authorization: Bearer token
  ▼
/api/chat
  │
  ▼
Flask /get
  │
  ▼
Supabase token validation
  │
  ▼
Pinecone retrieval
  │
  ▼
Groq
  │
  ▼
Answer
  │
  ▼
SQLAlchemy
  │
  ▼
PostgreSQL

Guest chat

Browser
  │
  ├── nanoid guest session
  │
  ▼
localStorage
  │
  ▼
Frontend
  │
  │ X-Guest-Session-Id
  ▼
Flask
  │
  ▼
Guest user mapping
  │
  ├── Pinecone
  ├── Groq
  └── Database

18. Current Architecture Strengths

The current implementation already has several useful production-oriented characteristics:

Separation between frontend and backend

Dedicated authentication module

Dedicated schema validation

Repository layer for database access

SQLAlchemy ORM

Alembic migration support

PostgreSQL support

Dockerized backend

Lazy RAG initialization

External vector database

Configurable model/retrieval parameters

CORS allowlisting

Security headers

Health endpoint

Authentication rate limits

Backend tests

Environment examples

Conversation persistence

19. Architecture Gaps

The following capabilities are not currently implemented or require further hardening.

Not currently implemented: distributed rate limiting

The current rate limiter is process-local.

Recommended production implementation:

Redis-backed rate limiting

API gateway/WAF limits

separate user/IP/token limits

stricter /get limits

Not currently implemented: RAG source citations

Documents retain source and page metadata during ingestion, but the /get response currently returns only the generated answer.

Recommended:

{
  "answer": "...",
  "sources": [
    {
      "source": "...",
      "page": 12
    }
  ]
}

Not currently implemented: automated indexing

store_index.py is a manually executed ingestion script.

Recommended:

versioned knowledge-base releases

ingestion job

document checksum tracking

duplicate prevention

reindexing strategy

CI/CD or scheduled ingestion where appropriate

Not currently implemented: formal LLM evaluation

Recommended:

retrieval evaluation

groundedness evaluation

medical safety evaluation

hallucination tests

regression datasets

prompt regression tests

Not currently implemented: frontend automated tests

A dedicated frontend test suite was not verified in the current repository.

Recommended:

Vitest

React Testing Library

route/component tests

chat flow tests

authentication state tests

20. Architecture Decision Principles

Future changes should preserve the following boundaries:

Frontend must not receive backend secrets.

Pinecone credentials must remain server-side.

Groq credentials must remain server-side.

Supabase publishable credentials may be used client-side according to Supabase's model, but privileged service-role credentials must never be bundled into the frontend.

Database access remains server-side.

RAG ingestion should remain separate from request-time generation.

Medical safety controls should be treated as defense-in-depth rather than as a substitute for clinical validation.

Production infrastructure should be reproducible.

API behavior should be documented whenever it changes.

Changes to the RAG knowledge base should be versioned and auditable.
