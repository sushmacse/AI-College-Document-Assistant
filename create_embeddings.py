from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Students must maintain at least 75% attendance.",
    "Hostel fee is Rs.45000 per year.",
    "Library opens from 9 AM to 6 PM."
]

embeddings = model.encode(documents)

print("\nEmbedding Shape:", embeddings.shape)

print("\nFirst 10 values of first embedding:")

print(embeddings[0][:10])