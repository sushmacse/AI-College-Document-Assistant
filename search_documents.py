from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection("college_documents")

query = input("Ask a Question: ")

embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[embedding],
    n_results=3
)

print("\nTop Results:\n")

for doc in results["documents"][0]:
    print(doc)
    print("-" * 50)
    