# Contributing to MediCore

Thank you for your interest in contributing to MediCore.

MediCore is an AI-powered medical information assistant built with a Flask backend, React/TanStack frontend, Supabase authentication, PostgreSQL/SQLite persistence, Pinecone vector search, Hugging Face embeddings, and Groq-based LLM inference.

Contributions are welcome, but changes to the RAG pipeline, authentication, database schema, and security-sensitive code require additional care because they can affect application correctness, privacy, reliability, and medical safety.

---

## Before You Start

Before opening an issue or pull request:

1. Search existing issues and pull requests.
2. Check the current `README.md`.
3. Run the relevant tests and validation commands locally.
4. Keep changes focused on one problem.
5. Do not commit secrets, credentials, personal data, or private medical information.
6. Do not introduce changes that make the application appear to provide professional medical diagnosis or treatment.

For larger architectural changes, open an issue first so the approach can be discussed before implementation.

---

# Development Environment

## Backend

The backend requires Python 3.10.

Create and activate a virtual environment:

```bash
cd backend

python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

On Windows:

```powershell
copy .env.example .env
```

Do not commit `.env`.

---

## Frontend

The frontend is located in:

```text
fronted/
```

Install dependencies:

```bash
cd fronted
npm install
```

Create the local environment file:

```bash
cp .env.example .env
```

Configure at minimum:

```env
VITE_SUPABASE_URL=
VITE_SUPABASE_PUBLISHABLE_KEY=
VITE_BACKEND_URL=http://127.0.0.1:1819
```

The repository currently contains both `package-lock.json` and Bun-related files. Unless a change specifically requires Bun, use npm for normal development so dependency installation remains consistent with the existing package-lock file.

---

# Repository Structure

```text
Medical-chat-bot-AI/
│
├── backend/
│   ├── app.py
│   ├── store_index.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── migrations/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   └── data/
│
├── fronted/
│   ├── src/
│   ├── supabase/
│   ├── package.json
│   └── render.yaml
│
└── documentation files
```

---

# Backend Development

The Flask application is defined in:

```text
backend/app.py
```

Important backend modules include:

```text
backend/src/
├── auth.py
├── database.py
├── helper.py
├── models.py
├── prompt.py
├── rate_limit.py
├── repositories.py
├── schemas.py
└── supabase_email.py
```

When modifying backend code:

- keep route handlers focused on HTTP concerns;
- keep database operations inside repository/database layers;
- validate request data before performing external operations;
- do not expose provider credentials;
- preserve authentication checks;
- preserve guest-session isolation;
- handle external service failures explicitly;
- avoid logging sensitive information.

---

# RAG Development

The RAG pipeline uses:

```text
Hugging Face embeddings
        ↓
Pinecone
        ↓
LangChain retrieval
        ↓
Groq LLM
```

The main indexing script is:

```text
backend/store_index.py
```

The current medical knowledge source is stored under:

```text
backend/data/
```

Changes to any of the following require particular care:

- embedding model
- embedding dimensions
- chunk size
- chunk overlap
- Pinecone index configuration
- retriever configuration
- system prompt
- LLM model
- temperature
- maximum output tokens

Changing the embedding model generally requires rebuilding the vector index.

Do not change the medical system prompt solely to make responses sound more confident. The system should remain appropriately cautious about uncertainty and medical limitations.

---

# Database Changes

Database models are defined under:

```text
backend/src/models.py
```

Database configuration is handled through:

```text
backend/src/database.py
```

Migrations are managed with Alembic.

Create a migration:

```bash
cd backend

alembic revision --autogenerate -m "describe database change"
```

Apply migrations:

```bash
alembic upgrade head
```

Do not modify an already-applied migration to change production schema history.

Instead, create a new migration.

Every pull request containing a schema change must explain:

- what changed;
- why it changed;
- whether existing data is affected;
- whether the migration is reversible;
- whether indexes or constraints were added;
- whether application code must be deployed before or after the migration.

---

# Authentication Changes

Authentication uses Supabase Auth.

Relevant frontend code is under:

```text
fronted/src/integrations/supabase/
fronted/src/lib/auth.ts
```

Backend authentication logic is under:

```text
backend/src/auth.py
backend/src/supabase_email.py
```

Authentication changes must preserve:

- email verification requirements;
- bearer-token validation;
- authenticated user isolation;
- guest-session isolation;
- password recovery behavior;
- allowed redirect-origin validation;
- backend session synchronization.

Never implement authentication by trusting user-provided identifiers without validating the authenticated identity.

---

# API Changes

The primary backend endpoints include:

```text
GET    /health
POST   /get

GET    /api/auth/me
POST   /api/auth/session
POST   /api/auth/logout
GET    /api/auth/email-settings
POST   /api/auth/send-otp
POST   /api/auth/send-recovery

GET    /api/conversations
POST   /api/conversations
GET    /api/conversations/<conversation_id>/messages
PATCH  /api/conversations/<conversation_id>
DELETE /api/conversations/<conversation_id>
```

When changing an API:

1. Update backend validation.
2. Update the frontend client.
3. Update tests.
4. Update the root README if the public behavior changes.
5. Document breaking changes in `CHANGELOG.md`.

Avoid changing response formats unnecessarily.

---

# Frontend Development

The frontend uses React and TanStack Start.

Routes are located under:

```text
fronted/src/routes/
```

Reusable components are located under:

```text
fronted/src/components/
```

Shared application logic is located under:

```text
fronted/src/lib/
```

When changing frontend behavior:

- preserve keyboard accessibility;
- preserve loading and error states;
- avoid exposing backend credentials;
- keep API calls centralized where practical;
- do not duplicate authentication logic unnecessarily;
- verify both authenticated and guest flows.

---

# Testing

Backend tests:

```bash
cd backend
pytest -q
```

Compile check:

```bash
python -m compileall app.py store_index.py src
```

Frontend lint:

```bash
cd fronted
npm run lint
```

Frontend production build:

```bash
npm run build
```

A pull request should not be considered ready if an existing test or build fails because of the change.

---

# Testing External Services

Tests should not require live production credentials unless the test is explicitly an integration test.

Avoid making normal unit tests dependent on:

- Groq API availability;
- Pinecone availability;
- Supabase email delivery;
- production databases.

External integrations should be mocked or isolated where appropriate.

Integration tests should clearly document which external services they require.

---

# Pull Requests

A pull request should contain:

### Summary

Explain what changed.

### Motivation

Explain why the change is needed.

### Implementation

Explain important implementation decisions.

### Testing

List the commands used to validate the change.

Example:

```text
pytest -q
npm run lint
npm run build
```

### Risk

Mention anything that could affect:

- authentication;
- database schema;
- RAG quality;
- security;
- API compatibility;
- deployment.

---

# Pull Request Checklist

Before submitting:

- [ ] The change has a clear purpose.
- [ ] Existing functionality was not unnecessarily changed.
- [ ] Backend tests pass.
- [ ] Frontend lint passes.
- [ ] Frontend build passes.
- [ ] Database migrations are included when required.
- [ ] API behavior is documented when changed.
- [ ] No secrets are committed.
- [ ] No personal or medical data is included in test fixtures.
- [ ] Authentication and authorization behavior has been checked.
- [ ] Guest-session behavior has been checked if affected.
- [ ] RAG behavior has been checked if retrieval/prompt/model configuration changed.
- [ ] Security implications have been considered.
- [ ] `CHANGELOG.md` has been updated for user-visible changes.

---

# Commit Messages

Use concise commit messages that describe the change.

Recommended format:

```text
type: short description
```

Examples:

```text
feat: add conversation export
fix: prevent unauthorized conversation access
fix: handle Pinecone timeout
refactor: separate conversation repository logic
test: add auth session coverage
docs: update deployment instructions
chore: update backend dependencies
```

Recommended types:

```text
feat
fix
refactor
test
docs
chore
perf
security
```

---

# Security-Sensitive Changes

If a change affects:

- authentication;
- authorization;
- session handling;
- secrets;
- CORS;
- redirect validation;
- rate limiting;
- database access control;
- user data;
- medical safety behavior;

review `SECURITY.md` before opening the pull request.

Do not publicly disclose a newly discovered vulnerability through a normal GitHub issue.

---

# Medical Safety

MediCore is an informational AI application.

Contributors must not introduce functionality that presents generated output as:

- a confirmed diagnosis;
- a guaranteed medical recommendation;
- a replacement for emergency care;
- a prescription;
- a professional clinical judgment.

Changes affecting medical safety should include tests covering unsafe or ambiguous inputs where practical.

---

# Documentation

Update documentation when changing:

- installation;
- environment variables;
- API behavior;
- deployment;
- architecture;
- authentication;
- database schema;
- RAG configuration;
- user-visible functionality.

Documentation should describe the behavior that actually exists in the code.

Do not document planned functionality as if it were already implemented.

---

# Code Review Expectations

Reviewers should focus on:

1. Correctness
2. Security
3. Data isolation
4. Error handling
5. Maintainability
6. Test coverage
7. Performance
8. API compatibility
9. Medical safety
10. Operational impact

A small, well-tested change is preferable to a large refactor without a clear migration strategy.

---

# License

By contributing code or documentation to this repository, you agree that your contribution may be distributed under the license applicable to the repository.

If your contribution contains third-party material, make sure its license permits redistribution and clearly document the source.

---

# Questions

For normal development questions and project discussions, use the project's GitHub issue/discussion mechanisms.

For security vulnerabilities, follow the process described in `SECURITY.md`.

Thank you for helping improve MediCore.
