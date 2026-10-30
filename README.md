# Adaptive RAG Evaluation & Optimization Framework

A scientific, modular, and reproducible framework for systematically benchmarking, evaluating, and optimizing Retrieval-Augmented Generation (RAG) pipelines.

---

## 🎯 Project Overview

Most RAG systems are developed heuristically without quantitative measurement of how chunking size, retrieval strategy, or reranking affects retrieval quality and system latency.

This framework is designed as a rigorous Machine Learning and Information Retrieval experimentation platform that allows researchers and engineers to:
1. **Benchmark Multiple Retrieval Paradigms:** Sparse lexical search (BM25), dense semantic search (Sentence Transformers + FAISS), and hybrid fusion (Reciprocal Rank Fusion).
2. **Optimize Pipeline Hyperparameters:** Dynamically search for the best chunk sizes, overlap ratios, top-k parameters, and fusion weights.
3. **Evaluate with Standard IR Metrics:** Quantitatively measure ranking performance using Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG).
4. **Profile Latency & Computational Trade-offs:** Measure per-stage latencies (chunking, embedding, index lookup, reranking, generation) to determine practical efficiency vs. accuracy trade-offs.
5. **Generate Grounded Answers:** Produce verifiable LLM responses with source citations linked to exact document chunks.

---

## 🏗️ Architecture & Modules

The repository is structured into modular, test-driven packages:

```
rag-evaluation-framework/
├── configs/            # Experiment and pipeline configuration files (YAML)
├── data/               # Raw corpora, processed chunks, and evaluation datasets
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── src/
│   └── rag_framework/  # Core package
│       ├── chunking/   # Text chunking strategies (fixed, recursive, sliding window)
│       ├── retrieval/  # BM25, FAISS dense retrieval, RRF hybrid fusion, rerankers
│       ├── evaluation/ # Ranking metrics (Recall, Precision, MRR, nDCG) and latency profiler
│       ├── generation/ # Grounded generator with explicit citation attribution
│       └── api/        # FastAPI REST endpoints
├── tests/              # Comprehensive test suites
│   ├── unit/           # Unit tests for individual components
│   └── integration/    # End-to-end pipeline and evaluation tests
├── ui/                 # Streamlit interactive dashboard
├── pyproject.toml      # Project metadata, build configuration, and dependencies
└── README.md
```

---

## 🔬 Core Components

* **Configurable Chunking:** Parameterized token/character chunking with configurable overlap and sentence boundary preservation.
* **Sparse Lexical Search (BM25):** Term frequency / inverse document frequency scoring for exact matches and keyword queries.
* **Dense Semantic Search (FAISS):** Embedding-based retrieval powered by Sentence Transformers.
* **Hybrid Fusion (Reciprocal Rank Fusion):** Rank-based fusion merging sparse and dense candidates without score distribution mismatch.
* **Cross-Encoder Reranking:** Deep context-aware scoring for top candidate reranking.
* **Evaluation Engine:** Full mathematical implementation of standard Information Retrieval (IR) metrics.
* **Grounded Answer Generation:** Structured prompt templates enforcing hallucination prevention and exact source citations.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.11+ (Current environment: Python 3.13.7)
* Git

### Environment Setup
```bash
# Clone or navigate to repository
cd rag-evaluation-framework

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

---

## 🧪 Testing

All modules are accompanied by unit tests. Once dependencies are installed:
```bash
pytest
```
