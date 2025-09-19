import os
import typing
from dotenv import load_dotenv
from flask import Flask, render_template, request

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = Flask(__name__, template_folder=".", static_folder=".")

# Load environment variables
load_dotenv()
API_KEY = os.environ.get("API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_KEY = os.environ.get("API_KEY") or API_KEY  # support prior naming

if API_KEY:
    os.environ["API_KEY"] = API_KEY

# Define the path for the FAISS index
INDEX_PATH = "medical-chat-bot"
DATA_DIR = "data/"

# Global variables for the model and retriever
groq_llm = None
retriever = None

def download_hugging_face_embeddings():
    """Download the Embeddings from HuggingFace."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdf_file(data_path: str):
    """Extract Data From the PDF files in a directory."""
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def filter_to_minimal_docs(docs: typing.List[Document]) -> typing.List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: typing.List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs

def text_split(extracted_data: typing.List[Document]):
    """Split the Data into Text Chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)

def initialize_model_and_retriever():
    """Initialize embeddings, FAISS retriever, and LLM model."""
    global groq_llm, retriever
    print("Initializing model and retriever...")

    if not API_KEY:
        raise RuntimeError("API_KEY (or API_KEY) not set in environment")

    embeddings = download_hugging_face_embeddings()

    # Ensure data dir exists
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    # Attempt to load PDFs (may be empty)
    extracted_data = load_pdf_file(DATA_DIR)
    if extracted_data:
        filter_data = filter_to_minimal_docs(extracted_data)
        text_chunks = text_split(filter_data)
    else:
        text_chunks = []

    # Create or load FAISS
    if not os.path.exists(INDEX_PATH):
        if not text_chunks:
            print("No PDFs found in data/ and no existing FAISS index. The app will start but retrieval will return no context.")
            # Create an empty index by adding a dummy doc to allow .as_retriever to exist
            # Better: delay creation until docs exist, but we keep it simple here.
            docsearch = FAISS.from_texts([""], embedding=embeddings)
        else:
            print("Creating new FAISS vector store...")
            docsearch = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
        docsearch.save_local(INDEX_PATH)
        print("FAISS vector store created and saved.")
    else:
        print("Loading existing FAISS vector store...")
        docsearch = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS vector store loaded.")

    groq_llm = ChatGroq(model="mixtral-8x7b-32768")
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})
    print("Model and retriever initialized.")

# Define the system prompt
system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

@app.route("/")
def home():
    return render_template("medical-chat-bot.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    global groq_llm, retriever
    user_input = request.form.get("msg", "").strip()
    if not user_input:
        return "Please enter a question."

    if groq_llm is None or retriever is None:
        return "Model not initialized. Please restart the server after initialization."

    question_answer_chain = create_stuff_documents_chain(groq_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        response = rag_chain.invoke({"input": user_input})
        answer = response.get("output_text", "Sorry, I couldn't generate an answer.")
    except Exception as e:
        answer = f"Error generating answer: {e}"

    return str(answer)

if __name__ == "__main__":
    initialize_model_and_retriever()  # initialize once before serving
    app.run(debug=True)
