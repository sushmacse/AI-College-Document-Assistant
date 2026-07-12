import os

import streamlit as st
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


# Page settings
st.set_page_config(
    page_title="AI College Document Assistant",
    page_icon="🎓"
)
st.markdown("""
<style>
.stApp {
    background-color: black;
    color: white;
}

h1, h2, h3 {
    color: white;
}

div[data-testid="stTextInput"] input {
    background-color: #222222;
    color: white;
}

.stButton > button {
    background-color: #00BFFF;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI College Document Assistant")
st.write("Ask questions from your college documents")



# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load database
db = Chroma(
    persist_directory="college_db",
    embedding_function=embeddings
)


retriever = db.as_retriever(
    search_kwargs={"k": 2}
)


# User input
question = st.text_input("Enter your question:")


question = st.chat_input("Ask a question about your college documents...")

if question:
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer only using the context below.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    st.subheader("Answer")
    st.write(response.text)