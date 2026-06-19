import logging
import os
import time

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    download_hugging_face_embeddings,
    filter_to_minimal_docs,
    load_pdf_file,
    text_split,
)


DEFAULT_INDEX_NAME = "medical-chatbot"
EMBEDDING_DIMENSION = 384
INDEX_READY_TIMEOUT_SECONDS = 120


load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def ensure_index(pc: Pinecone, index_name: str) -> None:
    if pc.has_index(index_name):
        logger.info("Using existing Pinecone index: %s", index_name)
        return

    logger.info("Creating Pinecone index: %s", index_name)
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )

    deadline = time.monotonic() + INDEX_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = pc.describe_index(index_name).status
        ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            return
        time.sleep(1)
    raise TimeoutError(f"Pinecone index was not ready after {INDEX_READY_TIMEOUT_SECONDS} seconds")


def main() -> None:
    pinecone_api_key = required_env("PINECONE_API_KEY")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    data_dir = os.getenv("PDF_DATA_DIR", "data")
    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)

    extracted_data = load_pdf_file(data=data_dir)
    minimal_docs = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(minimal_docs)
    embeddings = download_hugging_face_embeddings()

    pc = Pinecone(api_key=pinecone_api_key)
    ensure_index(pc, index_name)

    logger.info("Uploading %s chunks to Pinecone index %s", len(text_chunks), index_name)
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings,
    )
    logger.info("Indexing completed")


if __name__ == "__main__":
    main()
