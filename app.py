import os

from dotenv import load_dotenv
from google import genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


# Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable or add it to .env before running this app.")

client = genai.Client(api_key=api_key)


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database
db = Chroma(
    persist_directory="college_db",
    embedding_function=embeddings
)


# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 2}
)


# Ask question
question = input("Ask your question: ")


# Find relevant information
docs = retriever.invoke(question)


context = "\n\n".join(
    doc.page_content for doc in docs
)


# Send context to Gemini
prompt = f"""
You are a college document assistant.

Answer only using the given context.

Context:
{context}

Question:
{question}
"""


response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)


print("\nAnswer:")
print(response.text)