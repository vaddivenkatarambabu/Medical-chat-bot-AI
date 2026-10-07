import os
import time

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    document_chunk_id,
    download_hugging_face_embeddings,
    filter_to_minimal_docs,
    load_pdf_file,
    text_split,
)

DEFAULT_INDEX_NAME = "medical-chatbot"
DEFAULT_PINECONE_CLOUD = "aws"
DEFAULT_PINECONE_REGION = "us-east-1"
DEFAULT_PDF_DATA_DIR = "data"
PINECONE_DIMENSION = 384
PINECONE_METRIC = "cosine"
INDEX_READY_TIMEOUT_SECONDS = 120


class IndexingConfigurationError(RuntimeError):
    pass


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise IndexingConfigurationError(f"Missing required environment variable: {name}")
    return value


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _index_ready(description) -> bool:
    status = getattr(description, "status", None)
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


def _wait_for_index(pc: Pinecone, index_name: str) -> None:
    deadline = time.monotonic() + INDEX_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        description = pc.describe_index(index_name)
        if _index_ready(description):
            return
        time.sleep(2)

    raise TimeoutError(f"Pinecone index {index_name!r} was not ready in time")


def main() -> None:
    load_dotenv()

    pinecone_api_key = _required_env("PINECONE_API_KEY")
    index_name = _env("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    cloud = _env("PINECONE_CLOUD", DEFAULT_PINECONE_CLOUD)
    region = _env("PINECONE_REGION", DEFAULT_PINECONE_REGION)
    data_dir = _env("PDF_DATA_DIR", DEFAULT_PDF_DATA_DIR)

    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    extracted_data = load_pdf_file(data=data_dir)
    if not extracted_data:
        raise RuntimeError(f"No PDF documents found in {data_dir!r}")

    filtered_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filtered_data)
    if not text_chunks:
        raise RuntimeError(f"No text chunks were produced from {data_dir!r}")

    embeddings = download_hugging_face_embeddings()
    pc = Pinecone(api_key=pinecone_api_key)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        _wait_for_index(pc, index_name)

    ids = [document_chunk_id(document) for document in text_chunks]
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        ids=ids,
        index_name=index_name,
        embedding=embeddings,
    )

    print(f"Indexed {len(text_chunks)} chunks into Pinecone index {index_name!r}")


if __name__ == "__main__":
    main()
