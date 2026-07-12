import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

load_dotenv()


# Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable or add it to .env before running this app.")

client = genai.Client(api_key=api_key)


# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Connect ChromaDB

db = chromadb.PersistentClient(
    path="chroma_db"
)

collection = db.get_collection(
    name="college_documents"
)


# User question

question = input("Ask your college question: ")


# Convert question into embedding

query_embedding = embedding_model.encode(
    question
).tolist()


# Search database

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


context = "\n\n".join(
    results["documents"][0]
)


# Gemini prompt

prompt = f"""
You are an AI College Document Assistant.

Answer only using the given college documents.

Context:
{context}

Question:
{question}

Answer:
"""


# Gemini response

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)


print("\nAI Answer:")
print(response.text)