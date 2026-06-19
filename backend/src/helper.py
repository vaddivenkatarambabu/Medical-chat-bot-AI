from pathlib import Path
from functools import lru_cache
import os
from typing import Iterable, List, Union

# The app uses PyTorch sentence-transformer embeddings. Prevent Transformers
# from importing TensorFlow/Keras, which breaks on environments with Keras 3
# unless tf-keras is installed.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


PathLike = Union[str, Path]


def load_pdf_file(data: PathLike) -> List[Document]:
    """Load all PDF files from a directory."""
    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"PDF data directory does not exist: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"PDF data path is not a directory: {data_path}")

    loader = DirectoryLoader(
        str(data_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDF documents found in: {data_path}")
    return documents


def filter_to_minimal_docs(docs: Iterable[Document]) -> List[Document]:
    """Keep only metadata that is useful for retrieval citations."""
    minimal_docs: List[Document] = []
    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        metadata = {"source": source}
        if page is not None:
            metadata["page"] = page
        minimal_docs.append(Document(page_content=doc.page_content, metadata=metadata))
    return minimal_docs


def text_split(extracted_data: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(list(extracted_data))
    if not chunks:
        raise ValueError("No text chunks were created from the source documents.")
    return chunks


@lru_cache(maxsize=1)
def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
