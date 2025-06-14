# ingest_and_run.py
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# ==== Step 1: Ingest documents and persist Chroma DB ====

def build_vector_store():
    documents_path = "docs"
    persist_directory = "chroma"
    all_docs = []
    for filename in os.listdir(documents_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(documents_path, filename))
            all_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embedding=embedding, persist_directory=persist_directory)
    print("✅ Chroma DB created and persisted in:", persist_directory)

# ==== Step 2: Start Flask app ====

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Groq API key must be set in Render's environment variables
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    # Load vector store
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma", embedding_function=embedding)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key
    )

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    @app.route("/api/chat", methods=["POST"])
    def chat():
        try:
            user_message = request.json.get("message", "")
            if not user_message:
                return jsonify({"message": "Empty message"}), 400
            response = qa_chain.run(user_message)
            return jsonify({"message": response})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"message": f"Error: {str(e)}"}), 500

    return app

if __name__ == "__main__":
    build_vector_store()
    app = create_app()
    app.run(host="0.0.0.0", port=8000)
