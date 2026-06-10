
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from core.inference import get_hf_embedding,get_litellm,get_image_description
import os
new_llm = get_litellm('medium')
@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name="data_collection",
        embedding_function=get_hf_embedding(),
        persist_directory="./chroma_langchain_db",
    )
vector_store = get_vector_store()
my_prompt_template = ChatPromptTemplate.from_template("explain about the below question:- {input} based on the provided {context}")
combine_docs_chain = create_stuff_documents_chain(
    new_llm, my_prompt_template
)
retriver_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)


# Page configuration is now handled globally in app.py

# -----------------------------
# CONFIG
# -----------------------------

# -----------------------------
# LOAD VECTORSTORE
# -----------------------------


# -----------------------------
# UI
# -----------------------------

st.title("💬 Ask Documents")

st.markdown("""
Ask questions from your embedded PDFs.
""")

with st.sidebar:
    st.title("📚 RAG Dashboard")
    st.markdown("---")
    st.info("Semantic Search over PDFs")

# -----------------------------
# QUERY
# -----------------------------

query = st.text_input(
    "Ask your question"
)

# -----------------------------
# SEARCH
# -----------------------------

if query:
    st.subheader("💬 Answer")
    
    # We will capture the context documents here as the stream runs
    retrieved_context = []
    
    # Generator function to yield answer chunks from the retrieval chain
    def generate_response():
        for chunk in retriver_chain.stream({"input": query}):
            # Capture the context documents when they are yielded
            if "context" in chunk:
                retrieved_context.extend(chunk["context"])
            # Yield answer strings for the write_stream
            if "answer" in chunk:
                yield chunk["answer"]

    with st.spinner("Searching documents and generating answer..."):
        # st.write_stream consumes the generator and displays text dynamically
        answer = st.write_stream(generate_response)
        
    if not answer:
        st.warning("No relevant information found.")

    # Render the References UI AFTER the answer finishes streaming
    if retrieved_context:
        st.markdown("---")
        st.subheader("📑 References")
        
        # We use a set to avoid duplicating the same chunk if stream yielded it multiple times
        seen_contents = set()
        
        for i, doc in enumerate(retrieved_context):
            if doc.page_content in seen_contents:
                continue
            seen_contents.add(doc.page_content)
            
            meta = doc.metadata or {}
            file_name = meta.get('file_name', 'Unknown Document')
            page_num = meta.get('page_number', 'N/A')
            elem_type = meta.get('element_type', 'paragraph')
            
            with st.expander(f"Reference {i+1}: {file_name} (Page {page_num})"):
                st.markdown(f"**Element Type:** `{elem_type.upper()}`")
                st.write(doc.page_content)
                
                # Check if this context has an associated layout graphic (image, table crop)
                if meta.get('has_image') and meta.get('image_url'):
                    image_path = meta.get('image_url')
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"Extracted graphic from Page {page_num}", use_container_width=True)