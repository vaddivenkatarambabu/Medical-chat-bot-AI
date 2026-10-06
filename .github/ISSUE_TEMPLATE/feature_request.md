# Pull Request

## Summary

<!-- What does this PR change? -->

## Motivation

<!-- Why is this change needed? Link the issue when applicable. -->

## Related Issue

Closes #

## Change Type

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor
- [ ] Performance improvement
- [ ] Security improvement
- [ ] Documentation
- [ ] Dependency update
- [ ] Database migration
- [ ] CI/CD
- [ ] Other

## Affected Areas

- [ ] Backend
- [ ] Frontend
- [ ] Authentication
- [ ] Conversations
- [ ] Database
- [ ] RAG / retrieval
- [ ] LLM / prompts
- [ ] Supabase
- [ ] Pinecone
- [ ] Deployment
- [ ] Documentation
- [ ] GitHub Actions

## Implementation

<!-- Explain the important engineering decisions. Avoid repeating the diff line-by-line. -->

## Testing

### Backend

- [ ] `python -m pytest -q`
- [ ] `python -m ruff check .`
- [ ] `python -m ruff format --check .`
- [ ] `python -m compileall -q app.py store_index.py src migrations scripts`
- [ ] Alembic migration validated against PostgreSQL

### Frontend

- [ ] `npm run lint`
- [ ] `npx prettier --check .`
- [ ] `npx tsc --noEmit`
- [ ] `npm run build`

### Other Validation

<!-- Add integration tests, manual checks, screenshots, etc. -->

## Database Changes

- [ ] No database change
- [ ] Existing migration only
- [ ] New Alembic migration included
- [ ] Supabase migration included

If applicable, describe:

- Schema changes:
- Existing-data impact:
- Deployment ordering:
- Rollback strategy:

## API / Contract Changes

- [ ] No API contract change
- [ ] Request contract changed
- [ ] Response contract changed
- [ ] New endpoint
- [ ] Existing endpoint removed/deprecated

Describe any compatibility impact:

## RAG / AI Behavior

- [ ] No RAG or model behavior change
- [ ] Prompt changed
- [ ] Retrieval settings changed
- [ ] Embedding/indexing changed
- [ ] LLM/model changed
- [ ] Medical-safety behavior changed

If changed, explain:

- Expected behavior change:
- Evaluation performed:
- Known limitations:

## Security / Privacy

- [ ] No security-sensitive change
- [ ] Authentication/authorization reviewed
- [ ] Secrets/configuration reviewed
- [ ] CORS/redirect behavior reviewed
- [ ] Input validation reviewed
- [ ] User-data isolation reviewed
- [ ] Logging reviewed for sensitive data

Confirm:

- [ ] No credentials, tokens, or `.env` files were committed.
- [ ] No real patient or private medical data was added to tests, fixtures, screenshots, or logs.

## Medical Safety

- [ ] No medical-answering behavior changed
- [ ] Medical behavior reviewed
- [ ] Safety/disclaimer behavior reviewed
- [ ] Unsafe or ambiguous inputs considered

## Deployment / Operations

- [ ] No deployment impact
- [ ] Environment variables changed
- [ ] Render configuration changed
- [ ] Docker configuration changed
- [ ] Migration/deployment ordering required
- [ ] Rollback considerations documented

## Screenshots / Evidence

<!-- Include screenshots only when they materially help review the change. -->

## Final Checklist

- [ ] The PR has a focused scope.
- [ ] The implementation matches the existing architecture.
- [ ] Tests and quality checks pass.
- [ ] Documentation was updated where necessary.
- [ ] User-visible changes are reflected in `CHANGELOG.md`.
- [ ] Security and privacy implications were reviewed.
- [ ] Medical-safety implications were reviewed.
- [ ] No unnecessary dependency was introduced.
- [ ] No generated files were committed unintentionally.
