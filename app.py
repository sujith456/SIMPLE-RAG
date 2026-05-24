import streamlit as st

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="Document RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded" # Guarantees sidebar opens on load
)

def home_page():
    st.title("📚 Document RAG System")
    
    st.markdown("""
    Welcome to the **Document RAG System**.
    
    Use the sidebar to navigate:
    - **📤 Upload & Embed PDFs**: Build your document library and extract knowledge.
    - **💬 Ask Documents**: Perform semantic queries against your knowledge base.
    """)
    
    with st.sidebar:
        st.title("📚 Navigation")
        st.markdown("---")
        st.success("Select a page above to begin")

# Explicit navigation definition (Streamlit 1.36+ Best Practice)
pages = {
    "RAG System Menu": [
        st.Page(home_page, title="Home", icon="🏠"),
        st.Page("pages/upload_embed.py", title="Upload & Embed PDFs", icon="📤"),
        st.Page("pages/streamli_inference.py", title="Ask Documents", icon="💬"),
    ]
}

# Initialize and run the multi-page router
pg = st.navigation(pages)
pg.run()