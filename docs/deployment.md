MediCore Deployment Guide

1. Deployment Model

MediCore is deployed as separate frontend and backend services.

                    Internet
                       │
                       ▼
              ┌─────────────────┐
              │ Render Frontend │
              │ TanStack Start  │
              └────────┬────────┘
                       │
                       │ HTTPS
                       ▼
              ┌─────────────────┐
              │ Flask Backend   │
              │ Gunicorn/Docker │
              └───────┬─────────┘
                      │
          ┌───────────┼────────────┐
          │           │            │
          ▼           ▼            ▼
      Supabase     Pinecone      Groq
       Auth        Vector DB      LLM
          │
          ▼
     PostgreSQL

The exact cloud provider for the production backend/database is configurable and should be treated as deployment infrastructure rather than application logic.

2. Prerequisites

For a production deployment, provision:

GitHub repository

Frontend hosting

Backend hosting/container runtime

PostgreSQL database

Supabase project

Pinecone account/index

Groq API key

HTTPS

DNS/domain if using a custom domain

3. Backend Runtime Requirements

The backend currently targets:

Python >= 3.10,<3.11

The Dockerfile uses:

python:3.10-slim-bookworm

The backend starts with Gunicorn.

Typical command:

gunicorn --bind 0.0.0.0:$PORT --workers ${WEB_CONCURRENCY:-1} --timeout ${GUNICORN_TIMEOUT:-120} app:app

The Dockerfile already encapsulates this startup behavior.

4. Backend Environment Variables

The authoritative backend environment template is:

backend/.env.example

Required AI/vector credentials

PINECONE_API_KEY=
GROQ_API_KEY=

Without these, the RAG chat endpoint cannot generate responses.

Database

DATABASE_URL=
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=1800
DATABASE_AUTO_CREATE=

For local development, leaving DATABASE_URL empty causes the application to use:

instance/medicore.sqlite3

For production, PostgreSQL should be used.

Do not rely on automatic schema creation for production PostgreSQL.

Use Alembic migrations.

5. Supabase Configuration

The backend uses:

SUPABASE_URL
SUPABASE_PUBLISHABLE_KEY
SUPABASE_JWT_SECRET
SUPABASE_JWT_AUDIENCE
SUPABASE_JWT_ALGORITHMS

Typical values:

SUPABASE_JWT_AUDIENCE=authenticated
SUPABASE_JWT_ALGORITHMS=HS256

Depending on the deployment configuration, Supabase user verification may also use the Supabase Auth API.

6. Frontend/Auth URL Configuration

The backend uses:

FRONTEND_URL
CORS_ALLOWED_ORIGINS
AUTH_REDIRECT_ALLOWED_ORIGINS

These must contain the actual production frontend origin.

Example:

FRONTEND_URL=https://your-frontend.example.com

Do not use:

CORS_ALLOWED_ORIGINS=*

in a production deployment unless there is a deliberate security review for that configuration.

7. Pinecone Configuration

Current defaults:

PINECONE_INDEX_NAME=medical-chatbot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

The index must be compatible with the embedding model:

sentence-transformers/all-MiniLM-L6-v2

Dimension:

384

Metric:

cosine

Do not change the embedding model without rebuilding the vector index.

8. Retrieval and LLM Configuration

Current defaults:

RETRIEVER_K=3
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.2
GROQ_MAX_TOKENS=1024

These values are configurable.

Changes should be evaluated against:

retrieval precision

answer quality

latency

token usage

hallucination rate

medical safety

9. Flask/Gunicorn Configuration

Current variables include:

FLASK_DEBUG=0
FLASK_RUN_HOST=0.0.0.0
PORT=1819
WEB_CONCURRENCY=1
GUNICORN_TIMEOUT=120
LOG_LEVEL=INFO

Production must use:

FLASK_DEBUG=0

Do not enable Flask debug mode on a public production service.

10. Frontend Environment Variables

The frontend uses:

fronted/.env.example

Relevant variables:

VITE_SUPABASE_URL=
VITE_SUPABASE_PUBLISHABLE_KEY=
VITE_SUPABASE_PROJECT_ID=
VITE_BACKEND_URL=

The frontend must never receive privileged backend secrets.

In particular, do not expose:

GROQ_API_KEY
PINECONE_API_KEY
SUPABASE_SERVICE_ROLE_KEY
DATABASE_URL
SUPABASE_JWT_SECRET

to Vite/client-side environment variables.

11. Frontend Build

The frontend uses:

npm install
npm run build

The production start command is:

npm start

which runs:

srvx serve --prod --dir . --static dist/client --entry dist/server/server.js

The frontend is therefore a Node Web Service, not a simple static site.

12. Render Frontend Deployment

The current Render configuration is:

fronted/render.yaml

The configured service is:

type: web
runtime: node

Build:

npm install && npm run build

Start:

npm start

Node version:

22.19.0

The deployment uses:

VITE_SUPABASE_URL
VITE_SUPABASE_PUBLISHABLE_KEY
VITE_SUPABASE_PROJECT_ID
VITE_BACKEND_URL

as environment variables.

13. Backend Docker Deployment

The backend contains:

backend/Dockerfile

The image is based on:

python:3.10-slim-bookworm

The container exposes the application through the configured $PORT.

Build locally:

cd backend
docker build -t medicore-backend .

Run:

docker run --rm -p 1819:1819 --env-file .env medicore-backend

If the deployment platform provides a dynamic $PORT, configure the container/platform so the service binds to that port.

14. PostgreSQL Production Setup

Recommended production sequence:

Provision PostgreSQL.

Obtain the connection string.

Set DATABASE_URL.

Confirm the backend can connect.

Run Alembic migrations.

Start the application.

Verify:

GET /health?deep=1

Expected:

{
  "status": "ok",
  "database": "ok"
}

Production schema management should use:

alembic upgrade head

rather than relying on Base.metadata.create_all().

15. Pinecone Index Initialization

The repository contains:

backend/store_index.py

Run indexing from the backend directory:

cd backend
python store_index.py

The script:

Loads PDF files from PDF_DATA_DIR.

Extracts text.

Reduces metadata.

Splits documents.

Generates embeddings.

Creates the Pinecone index if required.

Uploads vectors.

The current source PDF is:

backend/data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf

16. Important Indexing Warning

store_index.py is an operational ingestion script.

It is not currently an automated migration system for the knowledge base.

Not currently implemented:

document versioning

document checksums

incremental indexing

duplicate detection

rollback

scheduled re-indexing

CI-triggered knowledge-base deployment

evaluation gates before publishing a new index

For a production medical system, these should eventually be treated as controlled data releases.

17. Supabase Production Configuration

Configure:

authentication provider

email verification

redirect URLs

frontend callback URLs

recovery URLs

email provider

production domain

The backend also validates allowed redirect origins.

Production redirect URLs should be explicit.

Avoid broad redirect allowlists.

18. Deployment Verification Checklist

After deployment verify:

Frontend

[ ] frontend loads
[ ] authentication UI loads
[ ] Supabase session works
[ ] chat request works
[ ] conversation list works
[ ] message history works

Backend

[ ] /
[ ] /health
[ ] /health?deep=1
[ ] POST /get
[ ] authentication endpoints
[ ] conversation endpoints

Infrastructure

[ ] PostgreSQL reachable
[ ] Alembic migrations applied
[ ] Pinecone index exists
[ ] Pinecone dimensions are correct
[ ] Groq credentials work
[ ] Supabase authentication works
[ ] CORS is restricted
[ ] HTTPS enabled

19. Production Troubleshooting

Backend starts but /get returns 503

Check:

PINECONE_API_KEY
GROQ_API_KEY

Also verify:

PINECONE_INDEX_NAME

and that the Pinecone index exists.

/get returns 500

Check backend logs for:

Failed to generate assistant response

Likely causes include:

Groq failure

Pinecone failure

embedding initialization failure

incompatible package versions

network connectivity

invalid vector index

Database health check fails

Check:

DATABASE_URL

Then verify:

alembic upgrade head

and database connectivity.

Authentication returns 401

Check:

SUPABASE_URL
SUPABASE_JWT_SECRET
SUPABASE_JWT_AUDIENCE
SUPABASE_JWT_ALGORITHMS

Also verify that the Supabase user's email is confirmed.

Browser CORS error

Check:

FRONTEND_URL
CORS_ALLOWED_ORIGINS

The exact browser origin must be included.

For example:

https://app.example.com

is different from:

https://www.example.com

Frontend chat proxy returns 500

The TanStack Start route:

fronted/src/routes/api/chat.ts

requires:

BACKEND_URL

or:

VITE_BACKEND_URL

Verify the server can reach the backend URL.

20. Production Hardening Still Required

Before treating the system as a high-scale production service, address:

distributed rate limiting

centralized logging

metrics

tracing

request IDs

alerting

database backups

disaster recovery

secret rotation

frontend automated tests

API contract tests

RAG evaluation

medical safety evaluation

source citations

controlled knowledge-base releases

backend infrastructure-as-code

These are recommendations, not claims that the current repository already implements them.
