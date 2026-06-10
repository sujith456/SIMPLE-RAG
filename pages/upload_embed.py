import pandas as pd 
import os
from core.inference import get_hf_embedding,get_litellm,get_image_description
from langchain_chroma import Chroma
from langchain_core.documents import Document
import sys
import os
import time
import json
import tempfile
from typing import List, Dict, Any
import pymupdf
from PIL import Image
from ultralytics import YOLO
import streamlit as st
import os
from pathlib import Path



from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import glob
from ingest.pdf_ingest import pdf_ingest
# Create upload folder if not exists
UPLOAD_FOLDER = "resources"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Page configuration is now handled globally in app.py

# -----------------------------
# CACHED VECTOR STORE INITIALIZATION
# -----------------------------
@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name="data_collection",
        embedding_function=get_hf_embedding(),
        persist_directory="./chroma_langchain_db",
    )

# -----------------------------
# PAGE-BY-PAGE VIEWER DIALOG
# -----------------------------
@st.dialog("📄 Document Content Viewer", width="large")
def view_document_dialog(pdf_name, chunks):
    st.markdown(f"### Document: `{pdf_name}`")
    
    # Group chunks by page number
    pages = {}
    for chunk in chunks:
        meta = chunk.get('metadata') or {}
        page_num = meta.get('page_number', 1)
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(chunk)
        
    sorted_pages = sorted(list(pages.keys()))
    
    if not sorted_pages:
        st.warning("⚠️ No page content found for this document.")
        if st.button("Close", use_container_width=True):
            st.rerun()
        return
        
    # Navigation tracking using session state
    state_key = f"dialog_page_{pdf_name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
        
    current_index = st.session_state[state_key]
    if current_index >= len(sorted_pages):
        current_index = 0
        st.session_state[state_key] = 0
        
    current_page = sorted_pages[current_index]
    
    # Navigation Controls
    nav_cols = st.columns([1, 2, 1])
    
    def go_prev():
        st.session_state[state_key] -= 1
        
    def go_next():
        st.session_state[state_key] += 1
        
    with nav_cols[0]:
        st.button("⬅️ Previous", disabled=(current_index == 0), key=f"dialog_prev_{pdf_name}", use_container_width=True, on_click=go_prev)
            
    with nav_cols[1]:
        st.markdown(f"<p style='text-align: center; font-size: 18px; margin: 0; padding-top: 5px;'><b>Page {current_page} of {sorted_pages[-1]}</b></p>", unsafe_allow_html=True)
        
    with nav_cols[2]:
        st.button("Next ➡️", disabled=(current_index == len(sorted_pages) - 1), key=f"dialog_next_{pdf_name}", use_container_width=True, on_click=go_next)
            
    st.markdown("---")
    
    # Display Chunks
    page_chunks = pages[current_page]
    
    # Sort chunks by y0 coordinate if coordinates are available to maintain reading flow
    def get_y_coord(ch):
        coords = ch.get('metadata', {}).get('bbox_coordinates')
        if coords and len(coords) >= 2:
            return coords[1]
        return 0
    
    page_chunks = sorted(page_chunks, key=get_y_coord)
    
    st.markdown(f"#### 📄 Page {current_page} Elements ({len(page_chunks)} chunks)")
    
    for idx, chunk in enumerate(page_chunks):
        meta = chunk.get('metadata') or {}
        elem_type = meta.get('element_type', 'paragraph')
        text_content = chunk.get('text', '')
        has_image = meta.get('has_image', False)
        image_url = meta.get('image_url') or meta.get('visual_crop_path')
        
        # Premium element wrapper styling
        if elem_type == 'table':
            st.markdown(f"**Element {idx+1} — 📊 TABLE**")
            st.info(text_content)
        elif elem_type == 'formula':
            st.markdown(f"**Element {idx+1} — 🧮 FORMULA**")
            st.warning(text_content)
        elif elem_type == 'heading':
            st.markdown(f"#### 🏷️ {text_content}")
        else:
            st.markdown(f"**Element {idx+1} — 📝 PARAGRAPH**")
            st.write(text_content)
            
        if has_image and image_url and os.path.exists(image_url):
            try:
                img = Image.open(image_url)
                st.image(img, caption=f"Visual crop of {elem_type} on Page {current_page}", use_container_width=True)
            except Exception as e:
                st.caption(f"[Visual Graphic: {image_url}] (Error loading: {e})")
                
        st.markdown("<div style='border-bottom: 1px dashed rgba(255,255,255,0.15); margin: 15px 0;'></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    if st.button("Close Viewer", key=f"dialog_close_{pdf_name}", use_container_width=True):
        st.rerun()


st.title("📤 Upload & Embed PDFs")

st.markdown("""
Upload PDF documents and generate embeddings.
""")

with st.sidebar:
    st.title("📚 RAG Dashboard")
    st.markdown("---")
    st.info("Upload documents and store embeddings in ChromaDB")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)
UPLOAD_FOLDER = "resources"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# -----------------------------
# PROCESS
# -----------------------------

if uploaded_files:

    st.write(f"Selected Files: {len(uploaded_files)}")

    if st.button("🚀 Generate Embeddings"):

        progress_bar = st.progress(0)

        vector_store = Chroma(
                collection_name="data_collection",
                embedding_function=get_hf_embedding(),
                persist_directory="./chroma_langchain_db",
            )

        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):

            # -----------------------------
            # SAVE PDF
            # -----------------------------

            filename = f"{uploaded_file.name}"

            save_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # -----------------------------
            # UI UPDATE
            # -----------------------------
            pdf_ingest(save_path,vector_store,progress_bar)

            st.success(
                f"✅ Processed {uploaded_file.name} "
            )

        st.toast("🎉 Embedding generation completed!")
        st.rerun()

# -----------------------------
# DOCUMENT LIBRARY & TABLE VIEW
# -----------------------------
st.markdown("---")
st.header("📂 Document Library & Embeddings Status")

vector_store = get_vector_store()
all_data = {"ids": [], "documents": [], "metadatas": []}
try:
    all_data = vector_store.get()
except Exception as e:
    st.error(f"Error fetching existing embeddings: {e}")

ids = all_data.get('ids', [])
documents = all_data.get('documents', [])
metadatas = all_data.get('metadatas', [])

chroma_pdfs = {}
for idx in range(len(ids)):
    meta = metadatas[idx] if metadatas else {}
    doc_text = documents[idx] if documents else ""
    
    file_name = meta.get('file_name') or meta.get('source')
    if not file_name:
        file_name = "Legacy / Unknown PDF"
        
    if file_name not in chroma_pdfs:
        chroma_pdfs[file_name] = []
        
    chroma_pdfs[file_name].append({
        "id": ids[idx],
        "text": doc_text,
        "metadata": meta
    })

pdf_files = glob.glob(os.path.join(UPLOAD_FOLDER, "*.[pP][dD][fF]"))
pdf_names = sorted(list(set([os.path.basename(f) for f in pdf_files])))

all_pdfs_info = {}
for name in pdf_names:
    all_pdfs_info[name] = {
        "in_folder": True,
        "chunks": chroma_pdfs.get(name, []),
        "chunk_count": len(chroma_pdfs.get(name, [])),
        "is_embedded": name in chroma_pdfs and len(chroma_pdfs[name]) > 0
    }

# Also include legacy/other PDFs in Chroma not present in folder
for name, chunks in chroma_pdfs.items():
    if name not in all_pdfs_info:
        all_pdfs_info[name] = {
            "in_folder": False,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "is_embedded": True
        }

if not all_pdfs_info:
    st.info("💡 No PDFs found in the system. Upload some PDFs above to start embedding!")
else:
    # Render table header
    cols = st.columns([3, 1.5, 1.5, 4])
    cols[0].markdown("**📄 PDF File Name**")
    cols[1].markdown("**📊 Chunks**")
    cols[2].markdown("**⚡ Status**")
    cols[3].markdown("**⚙️ Actions**")
    st.markdown("<hr style='margin: 5px 0 15px 0;'>", unsafe_allow_html=True)
    
    # Helper to clean up visual crops when deleting embeddings
    def cleanup_visual_assets(chunks_list):
        for chunk in chunks_list:
            meta = chunk.get('metadata') or {}
            img_path = meta.get('image_url') or meta.get('visual_crop_path')
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass

    # Render table rows
    for name, info in all_pdfs_info.items():
        cols = st.columns([3, 1.5, 1.5, 4])
        
        # 1. File name with display badge if not in folder
        if info['in_folder']:
            cols[0].markdown(f"**{name}**")
        else:
            cols[0].markdown(f"**{name}** *(Chroma only)*")
            
        # 2. Chunk count
        cols[1].write(f"{info['chunk_count']} chunks")
        
        # 3. Status Badge
        if info['is_embedded']:
            cols[2].markdown("<span style='color: #4CAF50; font-weight: bold;'>✅ Embedded</span>", unsafe_allow_html=True)
        else:
            cols[2].markdown("<span style='color: #FFC107; font-weight: bold;'>⚠️ Not Embedded</span>", unsafe_allow_html=True)
            
        # 4. Actions Grid Column
        with cols[3]:
            if info['is_embedded']:
                # Action layout for embedded files: View, Delete Embeddings, and Delete Document (if local file exists)
                sub_cols = st.columns(3 if info['in_folder'] else 2)
                
                # 1. View Button
                if sub_cols[0].button("👁️ View", key=f"view_{name}", use_container_width=True):
                    view_document_dialog(name, info['chunks'])
                    
                # 2. Delete Embeddings Button
                if sub_cols[1].button("🗑️ Del Embeds", key=f"del_embed_{name}", use_container_width=True, help="Deletes embeddings from Chroma DB"):
                    ids_to_delete = [c['id'] for c in info['chunks']]
                    if ids_to_delete:
                        try:
                            # Clean up visual crop images from disk
                            cleanup_visual_assets(info['chunks'])
                            # Delete from Chroma
                            vector_store.delete(ids=ids_to_delete)
                            st.success(f"Deleted embeddings for {name}!")
                            st.toast(f"🗑️ Embeddings deleted for {name}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                            
                # 3. Delete Document Button (if local file exists)
                if info['in_folder']:
                    if sub_cols[2].button("❌ Del Doc", key=f"del_doc_{name}", use_container_width=True, help="Deletes local PDF and its embeddings"):
                        ids_to_delete = [c['id'] for c in info['chunks']]
                        try:
                            # Clean up visual crops
                            cleanup_visual_assets(info['chunks'])
                            # Delete from Chroma if it has embeddings
                            if ids_to_delete:
                                vector_store.delete(ids=ids_to_delete)
                            # Delete local file
                            file_path = os.path.join(UPLOAD_FOLDER, name)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            st.success(f"Deleted document {name}!")
                            st.toast(f"❌ Document & embeddings deleted for {name}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                # Action layout for non-embedded local files: Embed, and Delete Document
                sub_cols = st.columns(2)
                
                # 1. Embed Button
                if sub_cols[0].button("🚀 Embed", key=f"embed_{name}", use_container_width=True, help="Generate Chroma embeddings"):
                    progress_bar = st.progress(0)
                    save_path = os.path.join(UPLOAD_FOLDER, name)
                    try:
                        pdf_ingest(save_path, vector_store, progress_bar)
                        st.success(f"✅ Embedded {name}!")
                        st.toast(f"🎉 Embedding generation completed for {name}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
                # 2. Delete Document Button
                if sub_cols[1].button("❌ Del Doc", key=f"del_doc_{name}", use_container_width=True, help="Deletes local PDF"):
                    file_path = os.path.join(UPLOAD_FOLDER, name)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            st.success(f"Deleted document {name}!")
                            st.toast(f"❌ Document deleted for {name}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")