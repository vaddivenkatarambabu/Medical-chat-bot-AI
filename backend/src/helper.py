import hashlib
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_KNOWLEDGE_BASE_VERSION = "local"


def _metadata_value(value):
    if value is None:
        return None
    return str(value)


def normalize_document_metadata(doc: Document) -> dict:
    metadata = {
        "source": _metadata_value(doc.metadata.get("source")),
        "page": doc.metadata.get("page"),
        "knowledge_base_version": os.getenv(
            "KNOWLEDGE_BASE_VERSION",
            DEFAULT_KNOWLEDGE_BASE_VERSION,
        ),
    }

    page_label = doc.metadata.get("page_label")
    if page_label is not None:
        metadata["page_label"] = _metadata_value(page_label)

    return {key: value for key, value in metadata.items() if value is not None}


def document_chunk_id(doc: Document) -> str:
    source = _metadata_value(doc.metadata.get("source")) or "unknown-source"
    page = _metadata_value(doc.metadata.get("page")) or "unknown-page"
    chunk = _metadata_value(doc.metadata.get("chunk")) or "unknown-chunk"
    version = (
        _metadata_value(doc.metadata.get("knowledge_base_version"))
        or DEFAULT_KNOWLEDGE_BASE_VERSION
    )
    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{Path(source).name}:{version}:p{page}:c{chunk}:{digest}"


# Extract Data From the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only citation/provenance metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=normalize_document_metadata(doc),
            )
        )
    return minimal_docs


# Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    for index, chunk in enumerate(text_chunks):
        chunk.metadata["chunk"] = index
    return text_chunks


# Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )  # this model return 384 dimensions
    return embeddings
