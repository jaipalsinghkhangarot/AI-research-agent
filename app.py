import streamlit as st
from dotenv import load_dotenv
from agent import create_agent
from document_handler import extract_text_from_pdf
from vector_store import store_text, query_vector_store

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant Agent")

agent = create_agent()

query = st.text_input("Ask a research question")

pdf = st.file_uploader("Upload a PDF (optional)", type="pdf")

if st.button("Run") and (query or pdf):
    context = ""

    if pdf:
        context = extract_text_from_pdf(pdf)
        store_text(pdf.name, context)  # save in Chroma (Gemini embeddings)
        st.subheader("Extracted PDF Summary")
        st.write(context[:1000] + "...")

    # Retrieve similar docs from memory
    similar_docs = query_vector_store(query)
    memory_context = "\n\n".join([doc.page_content for doc in similar_docs])

    final_query = f"{query}\n\nRelevant context from stored documents:\n{memory_context}"
    if context:
        final_query += f"\n\nNew PDF context:\n{context}"

    with st.spinner("Thinking..."):
        result = agent.run(final_query)
        st.success("Done!")
        st.markdown(result)
