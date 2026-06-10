
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