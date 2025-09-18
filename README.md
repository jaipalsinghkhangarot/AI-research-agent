AI Research Assistant

An AI-powered research assistant built with Streamlit, LangChain, ChromaDB, and Google Generative AI (Gemini embeddings).

It allows you to:

1. Upload PDFs and extract text
2. Store and search embeddings in ChromaDB
3. Ask research questions and retrieve context-aware answers
4. (Optional) Use audio transcription for voice queries

Features:
PDF ingestion & text chunking
Embeddings powered by Google Generative AI
Vector storage using Chroma
Conversational Q&A with LangChain
Simple and modern Streamlit UI

## Setup Instructions
1. Install dependencies:
pip install -r requirements.txt

2. Run the app:
streamlit run app.py or python -m streamlit run app.py

## Run Streamlit with LAN Access
streamlit run app.py --server.address 0.0.0.0 --server.port 8501