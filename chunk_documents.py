import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_FOLDER = "data"

documents = []

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(DATA_FOLDER, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
            documents.append(text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.create_documents(documents)

print(f"\nTotal Chunks Created: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print("="*60)
    print(f"Chunk {i+1}")
    print("="*60)
    print(chunk.page_content)