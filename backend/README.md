---
title: Medical Chat Bot AI
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Medical Chat Bot AI

Production-ready Flask retrieval-augmented medical assistant using LangChain,
Pinecone, Hugging Face sentence embeddings, and Groq chat models.

> Educational use only. This app does not replace a licensed clinician,
> emergency care, diagnosis, or professional medical advice.

## Architecture

- `templates/chat.html` serves the browser chat UI from Flask.
- `app.py` exposes `/`, `/health`, `/get`, and database-backed conversation APIs.
- `src/helper.py` loads PDFs, chunks text, and builds Hugging Face embeddings.
- `store_index.py` creates or reuses a Pinecone serverless index and uploads PDF chunks.
- Pinecone stores vectors. Groq generates answers from retrieved context.
- PostgreSQL stores users, hashed session metadata, conversations, and messages.
- SQLAlchemy manages ORM access, and Alembic manages database migrations.
- Supabase Auth sends OTP/verification/reset emails, hashes passwords, and
  issues JWT sessions.

The React frontend is served separately and calls the Flask API. Configure
`FRONTEND_URL` and `CORS_ALLOWED_ORIGINS` for the deployed frontend origin.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
copy .env.example .env
```

Fill `.env` with real values:

```env
PINECONE_API_KEY=...
GROQ_API_KEY=...
PINECONE_INDEX_NAME=medical-chatbot
DATABASE_URL=postgresql+psycopg://medicore:password@localhost:5432/medicore
SUPABASE_URL=...
SUPABASE_PUBLISHABLE_KEY=...
SUPABASE_JWT_SECRET=...
```

For local development only, you may omit `DATABASE_URL`; the backend will use
`instance/medicore.sqlite3` and auto-create tables. Production should always use
PostgreSQL and Alembic migrations.

## Database

Recommended production database: PostgreSQL.

Why PostgreSQL:

- It is the natural production target for the existing Supabase-based frontend.
- It supports durable relational chat history, user records, indexes, and
  transactional writes.
- SQLAlchemy and Alembic provide a mature ORM and migration workflow.

Apply migrations:

```bash
alembic upgrade head
```

Create a new migration after model changes:

```bash
alembic revision --autogenerate -m "describe change"
```

Seed demo data:

```bash
python -m scripts.seed
```

## Authentication

Production authentication uses Supabase Auth:

- Sign-up starts with an email OTP.
- After OTP verification, the user creates a password.
- Sign-in uses email and password.
- Forgot-password sends a secure Supabase recovery email.
- Reset-password uses the Supabase recovery session.
- Protected saved-chat APIs require a verified email address.

Password hashes are not duplicated in the Flask application database. Supabase
stores password hashes securely in its internal auth schema. The Flask database
stores application-safe user metadata, verification status, last-login time, and
SHA-256 hashes of access tokens for session auditing/revocation.

## Build The Vector Index

Place source PDFs in `data/`, then run:

```bash
python store_index.py
```

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`, which
creates 384-dimensional vectors. If you change the embedding model, recreate the
Pinecone index with the matching dimension before indexing.

## Run Locally

```bash
python app.py
```

Open `http://127.0.0.1:1819/`.

Health check:

```bash
curl http://127.0.0.1:1819/health
```

## Production Deployment

Recommended platform: Render Web Service using Docker plus a managed
PostgreSQL database.

Why Render:

- It supports long-running Flask/Gunicorn services.
- It provides a platform `PORT` environment variable for web services.
- It provides managed PostgreSQL or can connect to Supabase/Postgres.
- `render.yaml` keeps service configuration in the repo.
- Pinecone remains the managed vector database for retrieval.

Render's docs state that web services must bind to `0.0.0.0` and should use the
platform `PORT` value. This repo's Dockerfile and Procfile do that.

### Render Steps

1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Web Service from the repo.
3. Use the included `render.yaml`, or create a Docker web service manually.
4. Add secret environment variables:

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
DATABASE_URL=your_postgres_connection_string
SUPABASE_URL=your_supabase_project_url
SUPABASE_PUBLISHABLE_KEY=your_supabase_anon_or_publishable_key
SUPABASE_JWT_SECRET=your_supabase_jwt_secret
```

5. Confirm non-secret variables:

```env
PINECONE_INDEX_NAME=medical-chatbot
RETRIEVER_K=3
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.2
GROQ_MAX_TOKENS=1024
FRONTEND_URL=https://your-frontend.example.com
CORS_ALLOWED_ORIGINS=https://your-frontend.example.com
DATABASE_AUTO_CREATE=0
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
WEB_CONCURRENCY=1
GUNICORN_TIMEOUT=120
FLASK_DEBUG=0
```

6. Run migrations:

```bash
alembic upgrade head
```

7. Deploy and verify:

```bash
curl https://your-service.onrender.com/health
curl https://your-service.onrender.com/health?deep=1
```

Expected response:

```json
{"status":"ok"}
```

## Docker

```bash
docker build -t medical-chat-bot-ai .
docker run --env-file .env -p 1819:1819 medical-chat-bot-ai
```

## Validation

```bash
python -m compileall app.py store_index.py src
python -m pytest -q
```

## Environment Variables

| Name | Required | Purpose |
| --- | --- | --- |
| `PINECONE_API_KEY` | yes | Pinecone vector database API key |
| `GROQ_API_KEY` | yes | Groq chat model API key |
| `DATABASE_URL` | production | PostgreSQL connection URL; SQLite fallback is development only |
| `DATABASE_POOL_SIZE` | no | SQLAlchemy connection pool size for PostgreSQL |
| `DATABASE_MAX_OVERFLOW` | no | Extra PostgreSQL connections beyond pool size |
| `DATABASE_POOL_TIMEOUT` | no | Seconds to wait for a pooled database connection |
| `DATABASE_POOL_RECYCLE` | no | Seconds before recycling pooled database connections |
| `DATABASE_AUTO_CREATE` | no | Set `1` only for local/dev schema auto-create |
| `SUPABASE_URL` | production | Supabase project URL for Auth user metadata lookup |
| `SUPABASE_PUBLISHABLE_KEY` | production | Supabase anon/publishable key for Auth user metadata lookup |
| `SUPABASE_JWT_SECRET` | production | Verifies Supabase Bearer tokens before persisting backend users/sessions |
| `SUPABASE_JWT_AUDIENCE` | no | Expected Supabase JWT audience |
| `SUPABASE_JWT_ALGORITHMS` | no | Allowed JWT algorithms |
| `PINECONE_INDEX_NAME` | no | Existing/indexed Pinecone index name |
| `RETRIEVER_K` | no | Number of retrieved chunks |
| `GROQ_MODEL` | no | Groq model name |
| `GROQ_TEMPERATURE` | no | Chat generation temperature |
| `GROQ_MAX_TOKENS` | no | Maximum answer tokens |
| `FRONTEND_URL` | no | Frontend origin allowed to call the Flask API |
| `CORS_ALLOWED_ORIGINS` | no | Comma-separated origins allowed for browser API calls |
| `PORT` | no | Local server port; hosting platforms usually set this |
| `WEB_CONCURRENCY` | no | Gunicorn worker count |
| `GUNICORN_TIMEOUT` | no | Gunicorn request timeout |
| `MAX_CONTENT_LENGTH_BYTES` | no | Request body size limit |
| `LOG_LEVEL` | no | Python logging level |

## Production Notes

- Keep `.env` and API keys out of git.
- Use `/health` for fast platform health checks. Use `/health?deep=1` to verify
  the database connection.
- Index documents before deployment or run `python store_index.py` locally with
  production Pinecone credentials.
- Password handling remains with Supabase Auth. The Flask backend stores users
  from verified JWT claims and stores only hashed session tokens.
- Add rate limiting before making the service public.

## Useful References

- Render environment variables: https://render.com/docs/environment-variables
- Render Flask deployment: https://render.com/docs/deploy-flask
- Render Docker deployment: https://render.com/docs/docker
