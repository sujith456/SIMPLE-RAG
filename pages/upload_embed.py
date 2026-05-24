import pandas as pd 
import os
from inference import get_hf_embedding,get_litellm,get_image_description
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
DEFAULT_MODEL_NAME = "yolov10n_best.pt"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.getcwd()+'\\'+path
OUTPUT_DIR= ensure_dir('images')
class DocumentNode:
    """Represents a structural element parsed from the PDF page."""
    def __init__(self, rect: pymupdf.Rect, element_type: str, text: str = "", score: float = 1.0):
        self.rect = rect                
        self.element_type = element_type  
        self.text = text
        self.score = score
        self.crop_path = None

    def intersects(self, other: 'DocumentNode') -> bool:
        return self.rect.intersects(other.rect)
_YOLO_MODEL_CACHE = None
DEFAULT_MODEL_NAME = "yolov10n_best.pt"  

_YOLO_MODEL_CACHE = None

def get_yolov10_model(model_name: str = DEFAULT_MODEL_NAME) -> YOLO:
    """
    Downloads the YOLOv10 DocLayNet model from Hugging Face if not present,
    caches it, and returns the loaded Ultralytics YOLO instance.
    """
    global _YOLO_MODEL_CACHE
    if _YOLO_MODEL_CACHE is not None:
        return _YOLO_MODEL_CACHE

    local_path =f'C:\\Users\\sujit\\OneDrive\\Documents\\GENAI-Learning\\Complete Generative AI Course\\RAG_APP\\{DEFAULT_MODEL_NAME}'
    parent_path = None

    target_path = None
    if os.path.exists(local_path):
        target_path = local_path
    else:
        raise FileNotFoundError("Could not find YOLOv10 and no local fallback model was found.")

    print(f"[Info] Loading model weights into Ultralytics YOLO engine...")
    _YOLO_MODEL_CACHE = YOLO(target_path)
    print(f"[Success] Model loaded. Class Names: {list(_YOLO_MODEL_CACHE.names.values())}")
    return _YOLO_MODEL_CACHE

def run_data_pipeline(
    pdf_path: str, 
    page_num: int, 
    whitelist_classes: List[str] = ["picture", "table", "formula"],
    conf_threshold: float = 0.45,
    margin_threshold: float = 0.07,  # Ignore top/bottom 7% of page (Headers/Footers)
    min_size_px: float = 30.0,        # Ignore tiny artifacts
    max_area_ratio: float = 0.85      # Ignore giant canvas page backdrops
) -> Dict[str, Any]:
    """
    1. Visual Layout Detection (YOLOv10 ONNX/CPU) identifies complex graphic wrappers.
    2. Multiple filters (Margin, Size, Class, Conf) screen out page numbers, headers, and noise.
    3. Structural Layout Tree matches YOLO visual boundaries with native digital text extractions.
    4. Coordinate Math subtraction removes duplicate text chunks that overlap with graphics.
    5. Returns unified, deduplicated RAG payloads.
    """
    start_time = time.time()
    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    
    page_width = page.rect.width
    page_height = page.rect.height
    page_area = page_width * page_height

    yolo_dpi = 150 
    crop_dpi = 200
    
    pix_yolo = page.get_pixmap(dpi=yolo_dpi)
    temp_img_path = os.path.join(tempfile.gettempdir(), f"page_{page_num+1}.png")
    pix_yolo.save(temp_img_path)
    
    model = get_yolov10_model()
    results = model(temp_img_path, verbose=False)
    
    # Tracking statistics for filters
    stats = {
        "total_detected": 0,
        "filtered_by_class": 0,
        "filtered_by_conf": 0,
        "filtered_by_margin": 0,
        "filtered_by_size": 0,
        "extracted_visuals": 0
    }
    
    # Map raw pixel YOLO coordinates back to native PDF coordinates
    # YOLO runs on the rendered 150 DPI image.
    scale_x = page_width / pix_yolo.w
    scale_y = page_height / pix_yolo.h

    visual_nodes: List[DocumentNode] = []
    
    # ------------------ PHASE 3: APPLY PIPELINE FILTERS ------------------
    for idx, box in enumerate(results[0].boxes):
        stats["total_detected"] += 1
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id].lower()
        
        # 1. Class Filter
        if class_name not in [c.lower() for c in whitelist_classes]:
            stats["filtered_by_class"] += 1
            continue
            
        # 2. Confidence Filter
        if conf < conf_threshold:
            stats["filtered_by_conf"] += 1
            continue
            
        px0, py0, px1, py1 = map(float, box.xyxy[0])
        
        # Convert to native PDF coordinates (Points)
        x0 = px0 * scale_x
        y0 = py0 * scale_y
        x1 = px1 * scale_x
        y1 = py1 * scale_y
        
        # 3. Geometric Margin Filter (Header/Footer Gate)
        # Discards elements located in the very top or bottom page margins
        in_header = y0 < (margin_threshold * page_height)
        in_footer = y1 > ((1 - margin_threshold) * page_height)
        
        if in_header or in_footer:
            stats["filtered_by_margin"] += 1
            continue
            
        width = x1 - x0
        height = y1 - y0
        area = width * height
        
        # Ignore extremely thin lines or tiny visual ticks
        if width < min_size_px or height < min_size_px:
            stats["filtered_by_size"] += 1
            continue
            
        # Ignore massive background rectangles representing the PDF page background itself
        if area > (max_area_ratio * page_area):
            stats["filtered_by_size"] += 1
            continue
            
        # Successfully passed all filters! Add to visual structures tree
        rect = pymupdf.Rect(x0, y0, x1, y1)
        visual_nodes.append(DocumentNode(rect, class_name, score=conf))
        stats["extracted_visuals"] += 1

    if os.path.exists(temp_img_path):
        try:
            os.remove(temp_img_path)
        except Exception:
            pass

    # ------------------ PHASE 4: EXTRACT TEXT & BUILD LAYOUT TREE ------------------
    text_blocks = page.get_text("blocks")
    text_nodes: List[DocumentNode] = []
    
    for b in text_blocks:
        tx0, ty0, tx1, ty1, text, block_no, block_type = b
        rect = pymupdf.Rect(tx0, ty0, tx1, ty1)
        cleaned_text = text.strip()
        
        if not cleaned_text or block_type == 1:
            continue
            
        # Filter headers/footers geometrically from native text stream as well
        if ty0 < (margin_threshold * page_height) or ty1 > ((1 - margin_threshold) * page_height):
            continue
            
        # Classify Heading vs Paragraph
        if len(cleaned_text) < 50 and (cleaned_text.isupper() or cleaned_text[0].isdigit()):
            content_type = "heading"
        else:
            content_type = "paragraph"
            
        text_nodes.append(DocumentNode(rect, content_type, text=cleaned_text))

    # ------------------ PHASE 5: CROP VISUALS & DEDUPLICATE TEXT ------------------
    payloads = []
    zoom = crop_dpi / 72
    crop_matrix = pymupdf.Matrix(zoom, zoom)
    
    # Process visual elements: Crop and save at high resolution
    for idx, vis_node in enumerate(visual_nodes):
        clip = vis_node.rect + (-4, -4, 4, 4)  # Pad slightly
        clip.intersect(page.rect)
        
        crop_path = f"{OUTPUT_DIR}/s9_page_{page_num+1}_{vis_node.element_type}_{idx+1}.png"
        pix_crop = page.get_pixmap(matrix=crop_matrix, clip=clip)
        pix_crop.save(crop_path)
        vis_node.crop_path = crop_path
        
        payloads.append({
            "id": f"s9_page_{page_num+1}_visual_{idx+1}",
            "text": f"[Visual Reference Element: {vis_node.element_type.upper()} {idx+1} on page {page_num+1}]",
            "metadata": {
                "page_number": page_num + 1,
                "element_type": vis_node.element_type,
                "bbox_coordinates": [vis_node.rect.x0, vis_node.rect.y0, vis_node.rect.x1, vis_node.rect.y1],
                "visual_crop_path": crop_path,
                "confidence_score": vis_node.score,
                "has_image": True
            }
        })
        
    # Deduplicate Text Nodes overlapping with YOLO Visual boxes
    extracted_text_count = 0
    for idx, txt_node in enumerate(text_nodes):
        overlaps = False
        
        for vis_node in visual_nodes:
            if txt_node.intersects(vis_node):
                overlaps = True
                break
                
        if not overlaps:
            extracted_text_count += 1
            payloads.append({
                "id": f"s9_page_{page_num+1}_text_{idx+1}",
                "text": txt_node.text,
                "metadata": {
                    "page_number": page_num + 1,
                    "element_type": txt_node.element_type,
                    "bbox_coordinates": [txt_node.rect.x0, txt_node.rect.y0, txt_node.rect.x1, txt_node.rect.y1],
                    "visual_crop_path": None,
                    "confidence_score": 1.0,
                    "has_image": False
                }
            })
            
    doc.close()
    duration = time.time() - start_time
    
    return {
        "payloads": payloads,
        "stats": stats,
        "time_taken": duration,
        "extracted_text_count": extracted_text_count
    }

def merge_heading_and_small_paragraph(payloads):
    pending_heading=None
    merged_list=[]
    for payload in payloads:
        metadata=payload['metadata']
        payload_text = payload['text']
        element_type = payload['metadata']['element_type']
        if element_type in ['picture','table','formula']:
            if element_type == 'picture':
                if os.path.exists(metadata['visual_crop_path']):
                    payload['text']=payload_text = get_image_description(metadata['visual_crop_path'])
            if pending_heading:
                payload['text']=pending_heading+"\n\n"+ payload_text
                pending_heading=None
            merged_list.append(payload)
        else:
            if len(payload_text)<100 and  element_type in ["heading", "paragraph"]:
                if pending_heading:
                     pending_heading += "\n" + payload_text
                else:
                    pending_heading = payload_text
            elif element_type in ["heading", "paragraph"]:
                if pending_heading:
                    payload['text'] = pending_heading+payload_text
                    payload['metadata']['element_type']='section'
                    pending_heading=None
                merged_list.append(payload)
    if pending_heading:
        merged_list[-1]['text']+= merged_list[-1]['text']+pending_heading
    return merged_list


def pdf_ingest(pdf_path:str,vector_store,progress_bar):
    doc = pymupdf.open(pdf_path)
    total_pages_count = len(doc)
    doc.close()
    if total_pages_count == 0:
        progress_bar.progress(1.0, text="No pages found in document.")
        return
        
    print(f"\nProcessing all {total_pages_count} pages of the document {pdf_path}...")
    progress_bar.progress(0.0, text=f"Started processing {total_pages_count} pages...")
    for page_idx in range(total_pages_count):
        progress_bar.progress((page_idx + 1) / total_pages_count, text=f"Processing page {page_idx + 1} of {total_pages_count}...")
        print(f"\n[Page {page_idx+1}] Executing Ingestion Engine...")
        documents_list=[]
        final_chunks = []
        res = run_data_pipeline(
            pdf_path=pdf_path,
            page_num=page_idx,
            whitelist_classes=["picture", "formula"],
            conf_threshold=0.45,
            margin_threshold=0.07,  # Top/Bottom 7% ignored
            min_size_px=30.0,
            max_area_ratio=0.85
        )
        merged_output = merge_heading_and_small_paragraph(res["payloads"])
        for payload in merged_output:
            metadata = payload['metadata']
            pagenumber = metadata['page_number']
            has_image = metadata.get('has_image',False)
            image_url = metadata.get('visual_crop_path','')
            element_type = metadata.get("element_type", "unknown")
            filename = os.path.basename(pdf_path)
            documents_list.append(
                Document(
                    page_content=payload['text'],
                    metadata={
                        "page_number": pagenumber,
                        "element_type": element_type,
                        "has_image": has_image,
                        "image_url": image_url,
                        "file_name": filename,
                        "source": filename
                    }
                )
            )  
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " "]
            )
        for doc in documents_list:
            if doc.metadata["element_type"] in ["table", "picture", "formula"]:
                # Tables and images: NEVER split — they're atomic
                final_chunks.append(doc)
            elif len(doc.page_content) > 800:
                # Long sections: split with overlap
                final_chunks.extend(splitter.split_documents([doc]))
            else:
                # Normal merged sections: keep as-is
                final_chunks.append(doc)

        if final_chunks:
            vector_store.add_documents(final_chunks)





from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import glob

# Create upload folder if not exists
UPLOAD_FOLDER = "resources_pdf"
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
    st.info("Upload PDFs and store embeddings in ChromaDB")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)
UPLOAD_FOLDER = "resources_pdf"
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

# Scan local resources_pdf folder
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