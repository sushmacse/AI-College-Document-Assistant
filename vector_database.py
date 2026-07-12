import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB Client
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="college_documents"
)

# Read documents
documents = []

folder = "data"

for file in os.listdir(folder):
    if file.endswith(".txt"):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.create_documents(documents)

print("Total Chunks:", len(chunks))

# Store embeddings
for i, chunk in enumerate(chunks):

    embedding = model.encode(chunk.page_content).tolist()

    collection.add(
        ids=[str(i)],
        documents=[chunk.page_content],
        embeddings=[embedding]
    )

print("\nAll embeddings stored successfully!")