# Security Policy

## MediCore

MediCore is an AI-powered medical information application.

Because users may enter health-related information into the application, security and privacy issues are treated seriously.

This document explains how to report security vulnerabilities and the security expectations for contributors and deployments.

---

# Supported Versions

Security fixes are applied to the actively maintained version of the project.

| Version | Supported |
|---|---|
| `main` | Yes |
| Older commits/releases | No |

The `main` branch represents the current development version.

If a tagged release is introduced in the future, supported releases will be documented here.

---

# Reporting a Vulnerability

Please do **not** report security vulnerabilities through public GitHub issues.

A public issue can expose users and maintainers before a fix is available.

For security-sensitive findings, use GitHub's private vulnerability reporting/security advisory mechanism when it is enabled for the repository.

Repository:

[MediCore GitHub repository](https://github.com/vaddivenkatarambabu/Medical-chat-bot-AI?utm_source=chatgpt.com)

If private vulnerability reporting is unavailable, contact the repository maintainer privately through the contact mechanism associated with the GitHub account `vaddivenkatarambabu`.

Please avoid posting credentials, access tokens, personal information, or medical information in the report.

---

# What Should Be Reported Privately?

Examples include:

- authentication bypasses;
- authorization bypasses;
- access to another user's conversations;
- guest-session isolation failures;
- JWT validation weaknesses;
- session-token leakage;
- password-reset vulnerabilities;
- account takeover;
- CORS bypasses;
- unsafe authentication redirects;
- SQL injection;
- command injection;
- server-side request forgery;
- arbitrary file access;
- secret exposure;
- API key exposure;
- sensitive information disclosure;
- rate-limit bypasses with meaningful security impact;
- vulnerabilities allowing unauthorized modification or deletion of user data.

---

# What Does Not Usually Require a Security Report?

The following normally belong in a regular issue:

- UI bugs;
- spelling mistakes;
- documentation errors;
- normal API validation errors;
- feature requests;
- performance improvements without a security impact;
- ordinary dependency upgrade requests.

If a normal bug also creates a security or privacy risk, report it privately instead.

---

# Information to Include

A useful vulnerability report should contain:

```text
Title:
Affected component:
Affected endpoint/file:
Impact:
Steps to reproduce:
Expected behavior:
Actual behavior:
Proof of concept:
Affected versions/commits:
Suggested remediation:
```

Please include the minimum information required to reproduce the issue.

Do not include real user data.

For example, use:

```text
user@example.com
```

rather than a real user's email address.

---

# Severity

Security issues are evaluated according to practical impact.

Particular attention is given to vulnerabilities involving:

### Critical

- account takeover at scale;
- arbitrary code execution;
- exposure of application-wide credentials;
- cross-user access to sensitive conversation data.

### High

- authentication bypass;
- authorization bypass;
- access to another user's medical conversations;
- persistent secret exposure;
- serious injection vulnerabilities.

### Medium

- meaningful information disclosure;
- significant session weaknesses;
- security-control bypass with limited impact;
- rate-limit bypass affecting sensitive endpoints.

### Low

- limited information disclosure;
- defense-in-depth weaknesses;
- issues requiring unusual conditions to exploit.

Severity may be adjusted based on exploitability and real-world impact.

---

# Response Process

When a report is received, maintainers will:

1. Acknowledge receipt when practical.
2. Validate the reported behavior.
3. Determine affected components and versions.
4. Assess severity and impact.
5. Develop and test a fix.
6. Release the fix where appropriate.
7. Publish security guidance when disclosure is appropriate.

Do not assume that every report will result in a public security advisory.

---

# Secrets and Credentials

Never commit secrets to Git.

Examples include:

```text
GROQ_API_KEY
PINECONE_API_KEY
SUPABASE_JWT_SECRET
DATABASE_URL
database passwords
access tokens
private keys
service-role credentials
```

Local environment files such as:

```text
.env
```

must remain untracked.

Use the repository's `.env.example` files as templates.

---

# Supabase Security

The application uses Supabase Auth.

Authentication tokens must be treated as sensitive credentials.

Frontend code may use the Supabase publishable/client key intended for browser use.

Server-only credentials must never be exposed through:

```text
VITE_*
```

environment variables or frontend source code.

Backend authentication must continue validating authenticated requests rather than trusting frontend-provided user IDs.

---

# Database Security

Production deployments should use PostgreSQL rather than relying on the development SQLite database.

Production database credentials must be supplied through environment configuration or a managed secret store.

Database queries should continue using the existing SQLAlchemy/repository architecture.

Do not build SQL statements by concatenating untrusted user input.

Database migrations must be reviewed before deployment.

---

# Conversation Data

Users may enter sensitive information into chat conversations.

Contributors should therefore:

- avoid logging raw conversation content unnecessarily;
- avoid using real medical information in tests;
- avoid committing conversation exports;
- avoid copying production data into development environments;
- avoid exposing conversation IDs as authorization credentials;
- verify ownership before reading, modifying, or deleting conversations.

Guest-session identifiers should not be treated as equivalent to strong authentication.

---

# AI / RAG Security

The RAG system uses external services including Pinecone and Groq.

Do not send secrets or internal credentials to the LLM.

Be cautious when changing prompts or retrieval logic because retrieved documents and user input can influence model behavior.

Potential risks include:

- prompt injection;
- malicious content in indexed documents;
- data leakage through generated responses;
- incorrect retrieval;
- hallucinated medical information;
- unsafe instructions.

The application must not treat retrieved text as trusted executable instructions.

---

# Medical Safety

MediCore is not a clinical decision-making system.

Security reviews should consider not only conventional application security but also harmful model behavior.

Examples include:

- confidently fabricated diagnoses;
- unsafe medication instructions;
- failure to recognize emergencies;
- revealing private information through generated responses;
- prompt injection attempting to bypass safety behavior.

Changes to system prompts, retrieval behavior, model configuration, or medical data sources should be reviewed with these risks in mind.

---

# Rate Limiting

The backend includes rate limiting for selected endpoints.

The current implementation is process-local.

This means rate limiting should not be considered a complete distributed abuse-prevention mechanism when multiple backend workers or instances are deployed.

Production deployments should consider:

- Redis-backed rate limiting;
- API gateway limits;
- infrastructure-level request limits;
- authentication-specific abuse controls.

---

# CORS and Redirect Security

Production deployments should use explicit allowed origins.

Avoid:

```env
CORS_ALLOWED_ORIGINS=*
```

unless there is a documented reason and the associated security implications are understood.

Authentication redirects should only target explicitly trusted origins.

Never accept arbitrary redirect URLs from users.

---

# Dependency Security

Backend and frontend dependencies should be reviewed regularly.

Recommended practices:

- keep dependencies reasonably current;
- remove unused dependencies;
- review security advisories;
- use lock files consistently;
- enable Dependabot where appropriate;
- run dependency vulnerability checks in CI.

A dependency upgrade that changes RAG behavior, authentication behavior, or database behavior should be tested before deployment.

---

# Production Checklist

Before deploying MediCore to production:

- [ ] HTTPS enabled
- [ ] Debug mode disabled
- [ ] Production PostgreSQL configured
- [ ] Database migrations applied
- [ ] Secrets stored outside Git
- [ ] Supabase authentication configured
- [ ] JWT validation configured
- [ ] CORS restricted
- [ ] Authentication redirects restricted
- [ ] Pinecone credentials protected
- [ ] Groq credentials protected
- [ ] Logs reviewed for sensitive data
- [ ] Rate limiting configured
- [ ] Error monitoring enabled
- [ ] Database backups configured
- [ ] Dependency scanning enabled
- [ ] Secret scanning enabled
- [ ] Security headers verified
- [ ] Conversation authorization tested
- [ ] Guest-session isolation tested

---

# Disclosure

Responsible disclosure is preferred.

Please provide maintainers a reasonable opportunity to understand and address a vulnerability before publicly disclosing technical details.

Security fixes may be released before detailed vulnerability information is published.

Thank you for helping keep MediCore and its users safe.
