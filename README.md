# PDF RAG Application

A simple yet powerful **PDF-based RAG (Retrieval-Augmented Generation)** application built using:

* Streamlit for UI
* LangChain for orchestration
* PyMuPDF for PDF parsing
* YOLOv10 Nano for document layout detection

---

## Overview

Most basic RAG pipelines use standard document loaders that only extract plain text from PDFs.
However, real-world PDFs often contain:

* Tables
* Images
* Charts
* Formulas
* Complex layouts

These elements are important for understanding the document correctly.

This project implements a **multi-stage PDF extraction pipeline** to improve document understanding before embedding and retrieval.

---

## Features

* Upload and chat with PDFs
* Streamlit-based user interface
* LangChain-powered RAG pipeline
* Layout-aware PDF processing
* YOLOv10 Nano for layout detection
* Image and table region extraction
* Native digital text extraction using PyMuPDF
* Spatial filtering and coordinate-based processing
* Removal of unwanted icons, headers, and noisy elements
* Cleaner embeddings for better retrieval quality

---

# Pipeline Architecture

```text
PDF
 │
 ├── PyMuPDF Text Extraction
 │
 ├── YOLOv10 Nano Layout Detection
 │       ├── Tables
 │       ├── Images
 │       ├── Formulas
 │       └── Visual Regions
 │
 ├── Spatial Filtering
 │       ├── Remove icons
 │       ├── Remove noise
 │       ├── Remove overlapping text
 │       └── Filter unwanted regions
 │
 ├── Clean Structured Chunks
 │
 ├── Embeddings
 │
 └── RAG Retrieval + LLM Response
```

---

## Why This Approach?

Traditional PDF RAG systems usually miss important visual information because they rely only on text extraction.

This project improves retrieval quality by:

* Detecting document layouts visually
* Extracting structured content
* Preserving important tables and images
* Removing duplicated or noisy text
* Creating cleaner semantic embeddings

This leads to:

* Better context retrieval
* Better answers
* Reduced hallucinations

---

## Installation

```bash
git clone <your-repo-url>

cd project

pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Future Improvements

* OCR support for scanned PDFs
* Better table structure extraction
* Multi-modal embeddings
* Hybrid retrieval
* Caption-aware chunking
* Advanced metadata filtering
* GRAPH RAG implementation
* Agentic RAG
---

## Author

Built as a learning and experimentation project for advanced PDF RAG pipelines and document intelligence systems.
