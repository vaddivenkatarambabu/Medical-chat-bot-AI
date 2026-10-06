# Support

## MediCore

This document explains where to get help when working with or using MediCore.

MediCore is an AI-powered medical information application built with:

- React / TanStack Start
- Flask
- SQLAlchemy
- Alembic
- Supabase Auth
- PostgreSQL / SQLite
- Pinecone
- Hugging Face embeddings
- LangChain
- Groq

---

# Before Asking for Help

Start with the project README.

Check the following first:

```text
README.md
backend/README.md
fronted/README.md
backend/.env.example
fronted/.env.example
```

If the problem concerns authentication, also check the Supabase configuration and authentication flow documented by the project.

---

# Common Problems

## Backend Does Not Start

Verify:

```bash
python --version
```

The backend currently targets Python 3.10.

Then reinstall dependencies:

```bash
cd backend
pip install -r requirements-dev.txt
```

Check that the required environment variables are configured.

---

# `/get` Returns an Error

The chat endpoint depends on:

```text
Groq
Pinecone
Hugging Face embeddings
```

Check:

```text
GROQ_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
```

Also verify that the Pinecone index exists and contains the expected vectors.

---

# Frontend Cannot Reach Backend

Check:

```env
VITE_BACKEND_URL=http://127.0.0.1:1819
```

Then verify:

```text
http://127.0.0.1:1819/health
```

If `/health` does not respond, the problem is on the backend side.

If `/health` works but the browser reports a CORS error, verify:

```text
CORS_ALLOWED_ORIGINS
FRONTEND_URL
```

---

# Authentication Problems

MediCore uses Supabase Auth.

Verify:

```text
VITE_SUPABASE_URL
VITE_SUPABASE_PUBLISHABLE_KEY
```

on the frontend.

The backend also requires the appropriate Supabase configuration for token validation.

Verify:

```text
SUPABASE_URL
SUPABASE_PUBLISHABLE_KEY
SUPABASE_JWT_SECRET
SUPABASE_JWT_AUDIENCE
```

Do not post any of these secret values in an issue.

---

# Database Problems

For local development, verify the configured database URL.

For PostgreSQL:

```env
DATABASE_URL=postgresql+psycopg://...
```

Run migrations:

```bash
cd backend
alembic upgrade head
```

For migration-related problems, include:

```text
Python version
database type
migration command
migration revision
error message
```

Do not include database passwords or connection strings containing credentials.

---

# Pinecone Problems

Verify:

```text
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_CLOUD
PINECONE_REGION
```

If the embedding model has changed, verify that the Pinecone index was rebuilt with the matching vector dimension.

Do not paste Pinecone credentials into an issue.

---

# RAG Quality Problems

If the application runs but provides poor answers, provide:

```text
the question
the general expected behavior
whether retrieval appears relevant
whether the problem is reproducible
```

Do not submit real patient information or sensitive personal medical information.

Use synthetic examples instead.

Good:

```text
Question:
What are common symptoms associated with seasonal allergies?

Observed:
The answer discussed unrelated cardiovascular symptoms.
```

Bad:

```text
My real medical report says ...
```

---

# Reporting a Bug

Use a GitHub issue for normal bugs.

A useful bug report should contain:

```text
Environment:
OS:
Python version:
Node version:
Browser:
Application component:
Steps to reproduce:
Expected behavior:
Actual behavior:
Relevant logs:
```

Include the smallest reproducible example possible.

---

# Feature Requests

Before proposing a feature, check whether an existing issue already covers it.

A useful feature request should explain:

```text
Problem:
Who is affected:
Proposed behavior:
Why the current behavior is insufficient:
Potential implementation:
Potential risks:
```

Features involving medical recommendations, diagnosis, medication, or treatment should include additional discussion of safety implications.

---

# Security Issues

Do not create a public issue for a security vulnerability.

Follow:

```text
SECURITY.md
```

Examples include:

- authentication bypass;
- unauthorized conversation access;
- credential exposure;
- token leakage;
- SQL injection;
- arbitrary code execution;
- sensitive information disclosure.

---

# Pull Requests

If you have already identified the cause and implemented a fix, a pull request may be more appropriate than an issue.

Before opening one:

```bash
cd backend
pytest -q
```

and:

```bash
cd fronted
npm run lint
npm run build
```

Include the validation results in the pull request.

---

# What Support Cannot Guarantee

MediCore integrates several external services.

Problems may originate from:

- Supabase;
- Pinecone;
- Groq;
- Hugging Face;
- PostgreSQL;
- hosting infrastructure;
- DNS/network configuration.

The project maintainers cannot guarantee availability of third-party services.

When reporting an external-service failure, include enough information to determine whether the problem is application-specific or provider-specific.

---

# Medical Safety

MediCore is not an emergency service and does not provide professional medical diagnosis or treatment.

If someone is experiencing a medical emergency, they should seek appropriate emergency medical care rather than relying on this application.

Support requests should never contain unnecessary personal medical information.

---

# Maintainer Contact

For normal project questions, use GitHub issues or discussions where appropriate.

For security vulnerabilities, use the private reporting process described in `SECURITY.md`.

Repository:

[MediCore GitHub repository](https://github.com/vaddivenkatarambabu/Medical-chat-bot-AI?utm_source=chatgpt.com)
