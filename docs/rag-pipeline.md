# MediCore RAG Pipeline

## 1. Overview

MediCore uses Retrieval-Augmented Generation (RAG) to ground LLM responses in a project-managed medical knowledge base.

The pipeline separates:

1. Knowledge ingestion
2. Embedding generation
3. Vector storage
4. Retrieval
5. Prompt construction
6. LLM generation
7. Response persistence

The core implementation is located in:

```text
backend/store_index.py
backend/src/helper.py
backend/src/prompt.py
backend/app.py
```

---

# 2. Current Knowledge Source

The current repository includes:

```text
backend/data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf
```

The ingestion process is designed to load PDF files from:

```text
PDF_DATA_DIR
```

Default:

```text
data
```

The loader processes all matching:

```text
*.pdf
```

files in the configured directory.

---

# 3. Ingestion Architecture

```text
PDF files
   │
   ▼
DirectoryLoader
   │
   ▼
PyPDFLoader
   │
   ▼
LangChain Document objects
   │
   ▼
Metadata filtering
   │
   ▼
RecursiveCharacterTextSplitter
   │
   ▼
Text chunks
   │
   ▼
Hugging Face embeddings
   │
   ▼
384-dimensional vectors
   │
   ▼
Pinecone
```

---

# 4. PDF Loading

The implementation uses:

```python
DirectoryLoader
PyPDFLoader
```

The loader reads PDF files and creates LangChain `Document` objects.

Each document initially contains extracted text and loader metadata.

---

# 5. Metadata Normalization

The project deliberately reduces document metadata to:

```text
source
page
```

This is implemented in:

```text
backend/src/helper.py
```

The purpose is to keep useful provenance information without carrying unnecessary loader metadata into the vector store.

This metadata is important because it provides the foundation for future source citations.

---

# 6. Text Splitting

The project uses:

```text
RecursiveCharacterTextSplitter
```

Configuration:

```text
chunk_size = 500
chunk_overlap = 50
```

Therefore:

```text
500 characters maximum target chunk size
50 characters overlap
```

The overlap helps preserve context between neighboring chunks.

---

# 7. Why Chunking Exists

Sending an entire medical reference book to the LLM for every question would be:

- expensive
- slow
- impossible to fit reliably into the context window
- difficult to control

Instead, the system retrieves only relevant chunks.

Example:

```text
Question:
"What are symptoms of anemia?"

          ↓

Vector similarity search

          ↓

Relevant chunks about anemia

          ↓

LLM receives relevant context
```

---

# 8. Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through:

```python
HuggingFaceEmbeddings
```

Embeddings are normalized:

```python
encode_kwargs={"normalize_embeddings": True}
```

The vector dimension is:

```text
384
```

---

# 9. Embedding Consistency Requirement

The embedding model used during indexing must match the embedding model used during retrieval.

Current indexing:

```text
all-MiniLM-L6-v2
```

Current runtime:

```text
all-MiniLM-L6-v2
```

Changing this requires reindexing the Pinecone database.

For example, changing to a different embedding model without rebuilding the index would produce incompatible vector semantics.

---

# 10. Pinecone Index

Pinecone is the project's vector database.

The indexing script creates an index using:

```text
dimension = 384
metric = cosine
```

Default deployment configuration:

```text
cloud = aws
region = us-east-1
```

Index name:

```text
medical-chatbot
```

unless overridden by:

```text
PINECONE_INDEX_NAME
```

---

# 11. Index Creation

`backend/store_index.py` checks whether the configured Pinecone index exists.

If it does not exist, the script creates it.

The index is configured as a Pinecone serverless index.

The script waits for the index to become ready before uploading vectors.

---

# 12. Upload Process

After splitting and embedding:

```python
PineconeVectorStore.from_documents(...)
```

uploads the document chunks into Pinecone.

The vector records retain the relevant metadata from the source documents.

---

# 13. Runtime Retrieval

At runtime, the backend creates:

```python
PineconeVectorStore.from_existing_index(...)
```

The retriever is configured as:

```text
search_type = similarity
```

with:

```text
k = RETRIEVER_K
```

Default:

```text
3
```

Therefore each query retrieves the top three similarity matches by default.

---

# 14. Runtime RAG Flow

```text
User question
      │
      ▼
POST /get
      │
      ▼
Input validation
      │
      ▼
RAG chain
      │
      ├── Query embedding
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
Prompt construction
      │
      ▼
Groq LLM
      │
      ▼
Generated answer
```

---

# 15. LangChain Retrieval Chain

The backend creates:

```python
create_retrieval_chain(...)
```

with:

```python
create_stuff_documents_chain(...)
```

The retrieval chain combines:

- retriever
- document combination chain
- system prompt
- user input
- retrieved context

---

# 16. Prompt Construction

The system prompt is defined in:

```text
backend/src/prompt.py
```

It contains:

```text
system instructions
+
medical safety rules
+
retrieved context
```

The human message is:

```text
{input}
```

The context placeholder is:

```text
{context}
```

Conceptually:

```text
SYSTEM:
MediCore instructions
medical safety rules
retrieved reference context

USER:
current question
```

---

# 17. Medical Safety Instructions

The prompt instructs the model to:

- avoid diagnosis certainty
- avoid fabricated facts
- avoid unsupported statistics
- avoid personalized medication dosing
- recommend appropriate professional care
- identify emergency symptoms
- acknowledge insufficient information
- use retrieved medical context as the primary source when relevant

This is an important application-level safety layer.

However:

> A system prompt is not a complete medical safety mechanism.

It should be combined with testing, monitoring, controlled source material, input/output safeguards, and human review where appropriate.

---

# 18. LLM Provider

The current LLM provider is Groq.

LangChain integration:

```text
langchain_groq.ChatGroq
```

Default model:

```text
llama-3.3-70b-versatile
```

Configuration:

```text
GROQ_MODEL
GROQ_TEMPERATURE
GROQ_MAX_TOKENS
```

Defaults:

```text
temperature = 0.2
max_tokens = 1024
```

---

# 19. Why Low Temperature Is Used

The assistant is intended to provide grounded information rather than creative content.

A lower temperature generally makes output more deterministic.

The current configuration therefore favors:

```text
groundedness
consistency
predictability
```

over creative variation.

This does not guarantee factual correctness.

---

# 20. RAG Chain Caching

The RAG chain is cached:

```python
@lru_cache(maxsize=1)
```

This prevents the application from rebuilding the entire chain for every request.

Within a single Python process, the same initialized chain is reused.

If multiple Gunicorn workers are running, each worker maintains its own in-memory cache.

---

# 21. Response Persistence

After the answer is generated, the backend attempts to persist:

```text
user message
assistant answer
conversation
```

into the relational database.

This happens after generation.

A database failure does not currently cause the generated answer to be discarded.

---

# 22. Conversation History and RAG

The application stores previous messages.

However, the current RAG pipeline does not retrieve those messages during generation.

Therefore:

```text
Database conversation history
```

is currently used for UI/history purposes rather than as LLM memory.

**Not currently implemented:** conversation-aware RAG.

A future implementation could use:

```text
conversation summary
+
recent messages
+
current question
+
retrieved documents
```

but must carefully control token usage and prompt-injection risks.

---

# 23. Source Citations

The ingestion pipeline retains:

```text
source
page
```

metadata.

However, the `/get` API currently extracts only:

```python
response["answer"]
```

and returns the answer as plain text.

Therefore:

**Not currently implemented:** user-visible source citations.

Recommended future response:

```json
{
  "answer": "....",
  "sources": [
    {
      "source": "Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf",
      "page": 42
    }
  ]
}
```

This would improve transparency and make the RAG system auditable.

---

# 24. Current RAG Limitations

The current knowledge base is limited.

The repository currently contains one medical PDF source:

```text
Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf
```

Therefore the assistant should not be represented as a comprehensive medical knowledge system.

The source may not cover:

- current clinical guidelines
- newly approved treatments
- updated drug information
- regional medical guidance
- specialist-level clinical evidence

This limitation is important for the product's medical safety positioning.

---

# 25. RAG Improvements Recommended

### Knowledge-base versioning

Create explicit versions:

```text
medical-kb-v1
medical-kb-v2
```

### Source provenance

Store:

```text
document_id
document_version
source
page
chunk_id
ingestion_timestamp
checksum
```

### Evaluation

Add benchmark questions covering:

- retrieval correctness
- groundedness
- hallucination
- emergency advice
- refusal behavior
- source attribution

### Re-indexing

Build a repeatable ingestion pipeline with:

- checksums
- idempotency
- rollback
- validation
- monitoring

### Source-aware API

Return citations with answers.

---

# 26. Prompt Injection Considerations

Retrieved documents should not automatically be trusted as instructions.

Future RAG implementations should distinguish:

```text
retrieved knowledge
```

from:

```text
system instructions
```

Documents should be treated as untrusted data.

The system prompt should retain higher priority than retrieved content.

---

# 27. RAG Operational Checklist

Before publishing a new knowledge base:

```text
[ ] Source documents approved
[ ] Licensing reviewed
[ ] Documents parsed successfully
[ ] No unexpected empty documents
[ ] Chunk counts recorded
[ ] Embedding model verified
[ ] Pinecone dimension verified
[ ] Retrieval benchmark passed
[ ] Medical safety benchmark passed
[ ] Source metadata verified
[ ] New index tested
[ ] Rollback plan available
```

---

# 28. Current RAG Status

The current implementation is a functional application-level RAG pipeline.

It is suitable as a strong engineering project and prototype architecture.

For a medical product handling high-stakes decisions, it still requires additional:

- evaluation
- source governance
- monitoring
- provenance
- clinical review
- production safety controls
