#Overview
AI Research Assistant is a Streamlit-based application designed to help users interact with academic research papers through natural language queries.
The system processes uploaded PDF documents, performs semantic chunking, retrieves relevant information using hybrid search techniques, reranks the results using transformer-based models, and generates evidence-backed answers.
The project combines dense retrieval, sparse retrieval, and reranking methods to improve retrieval quality and contextual relevance.


#Features
PDF research paper ingestion
Semantic text chunking
Hybrid retrieval pipeline
FAISS dense vector search
BM25 sparse retrieval
Reciprocal Rank Fusion (RRF)
Cross-encoder reranking
Evidence-based answer generation
Confidence scoring
Multiple retrieval modes
Downloadable JSON results
Interactive Streamlit interface



#System Architecture
PDF Upload
    ↓
Text Extraction
    ↓
Cleaning & Preprocessing
    ↓
Semantic Chunking
    ↓
Embedding Generation
    ↓
FAISS Retrieval + BM25 Retrieval
    ↓
Reciprocal Rank Fusion
    ↓
Cross-Encoder Reranking
    ↓
Evidence Extraction
    ↓
Answer Generation
