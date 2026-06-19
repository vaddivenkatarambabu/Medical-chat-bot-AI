from setuptools import find_packages, setup

setup(
    name="medical-chat-bot",
    version="0.1.0",
    description="Flask RAG medical chatbot using Pinecone, LangChain, and Groq.",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
)
