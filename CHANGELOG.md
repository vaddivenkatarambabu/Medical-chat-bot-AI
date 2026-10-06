# Changelog

All notable changes to MediCore are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows Semantic Versioning when formal releases are created.

Until the first official release is tagged, unreleased development changes are recorded under `Unreleased`.

---

## [Unreleased]

### Added

#### Backend

- Flask-based API for medical question answering.
- Retrieval-augmented generation pipeline using LangChain.
- Pinecone vector retrieval.
- Hugging Face sentence-transformer embeddings.
- Groq LLM integration.
- Configurable retriever size.
- Configurable LLM model, temperature, and maximum output tokens.
- Request validation for chat payloads.
- Maximum chat-message length enforcement.
- Health endpoint.
- Deep database health check.
- Conversation persistence.
- Message persistence.
- Guest-session support.
- Authenticated user sessions.
- Conversation creation.
- Conversation listing.
- Conversation renaming.
- Conversation deletion.
- Conversation message retrieval.
- Backend session synchronization.
- Backend session revocation.
- Email OTP request handling.
- Password recovery email handling.
- Authentication redirect allowlisting.
- In-process API rate limiting.
- Security response headers.
- CORS configuration.
- Database connection pooling configuration.
- Alembic database migrations.
- Development seed script.
- Docker-based backend deployment support.

#### Frontend

- React-based chat interface.
- TanStack Start application structure.
- TanStack Router file-based routing.
- Authenticated chat routes.
- Guest chat flow.
- Supabase authentication integration.
- Email verification flow.
- Password sign-in flow.
- Password recovery flow.
- Password reset flow.
- Persistent Supabase browser sessions.
- Backend authentication session synchronization.
- Conversation sidebar.
- Conversation history.
- Conversation rename and delete operations.
- Medical disclaimer component.
- Suggested medical questions.
- Message loading states.
- Request cancellation.
- Request timeout handling.
- API error handling.
- Responsive chat interface.
- Theme support.
- Configurable backend URL.

#### Data / RAG

- Medical reference PDF ingestion pipeline.
- Recursive text chunking.
- Embedding generation.
- Pinecone vector index integration.
- Configurable retrieval count.
- Medical system prompt integration.

#### Development

- Backend development requirements.
- Pytest test suite.
- Frontend linting configuration.
- Frontend production build configuration.
- Environment templates.
- Render deployment configuration.
- Backend Docker configuration.

---

## [0.1.0] - Initial Development Baseline

This section represents the first documented project baseline.

### Included

- Full-stack MediCore application structure.
- Flask backend.
- React/TanStack frontend.
- Supabase authentication integration.
- Conversation persistence.
- Guest sessions.
- Pinecone-backed RAG pipeline.
- Groq LLM integration.
- Hugging Face embeddings.
- Alembic migrations.
- Backend automated tests.
- Docker support.
- Frontend deployment configuration.

> This version is a development baseline rather than a claim of clinical or production-grade medical software certification.

---

# Release Guidelines

When creating a new release, update this file before tagging the release.

Use categories where applicable:

```text
Added
Changed
Deprecated
Removed
Fixed
Security
```

Example:

```markdown
## [0.2.0] - 2026-XX-XX

### Added

- Added conversation export.

### Changed

- Improved Pinecone retrieval configuration.

### Fixed

- Fixed unauthorized conversation access.

### Security

- Added stricter authentication redirect validation.
```

Do not add a change to the changelog unless the change actually exists in the repository.

---

# Breaking Changes

Breaking changes must be clearly identified.

Examples include:

- API response changes;
- renamed environment variables;
- database migration requirements;
- authentication flow changes;
- removal of supported endpoints;
- incompatible frontend/backend versions;
- changes requiring Pinecone index recreation.

Use:

```markdown
### Changed

- **Breaking:** ...
```

when appropriate.

---

# Security Changes

Security fixes should be documented without publishing exploit details that could put users at unnecessary risk.

Example:

```markdown
### Security

- Fixed an authorization issue affecting conversation access.
```

Detailed vulnerability information should be handled according to `SECURITY.md`.

---

# RAG and Model Changes

Changes to the following should be documented because they can alter application behavior:

- LLM model;
- embedding model;
- Pinecone index;
- chunking configuration;
- retrieval parameters;
- system prompt;
- generation parameters.

Example:

```markdown
### Changed

- Updated the default Groq model.
- Increased retriever K from 3 to 5.
```

If a change requires rebuilding the Pinecone index, explicitly state that in the release notes.

---

# Database Changes

Database changes should mention migration requirements.

Example:

```markdown
### Changed

- Added conversation metadata fields.
- Added Alembic migration `20260610_0003`.
```

If deployment requires:

```bash
alembic upgrade head
```

state this in the release notes.

---

# Versioning

MediCore uses Semantic Versioning once formal releases are established:

```text
MAJOR.MINOR.PATCH
```

### MAJOR

Breaking API, database, or application changes.

### MINOR

Backward-compatible features.

### PATCH

Backward-compatible bug fixes and small improvements.

Until the project has formal release tags, development changes remain under:

```text
[Unreleased]
```

---

# Release Checklist

Before publishing a release:

- [ ] Tests pass.
- [ ] Frontend lint passes.
- [ ] Frontend build passes.
- [ ] Database migrations are reviewed.
- [ ] RAG changes are evaluated.
- [ ] Security-sensitive changes are reviewed.
- [ ] Environment variables are documented.
- [ ] Deployment instructions are current.
- [ ] Changelog is updated.
- [ ] Version is tagged.
- [ ] Release notes explain breaking changes.
