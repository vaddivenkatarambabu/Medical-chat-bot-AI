from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from groq import Groq
from langchain_community.vectorstores import FAISS

load_dotenv()


API_KEY=os.environ.get('API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["API_KEY"] = API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


extracted_data = load_pdf_file(data=r"C:\Users\venkatarambabuvaddi\Documents\Projects\Medical-chat-bot-AI\data")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = download_hugging_face_embeddings()

api_key = API_KEY
pc = Groq(api_key=api_key)


client = Groq(api_key=API_KEY)
completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
      {
        "role": "user",
        "content": "Whats is acne"
      }
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=False,
    stop=None
)

print(completion.choices[0].message)


docsearch = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings
)
