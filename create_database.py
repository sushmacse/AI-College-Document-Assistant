from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Load text document
loader = TextLoader("documents/college_rules.txt")

documents = loader.load()

print("Documents loaded:", len(documents))


# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create vector database
db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="college_db"
)

print("Vector database created successfully!")