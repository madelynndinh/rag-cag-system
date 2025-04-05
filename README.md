# RAG-CAG System

A modular framework integrating Retrieval-Augmented Generation (RAG) and Context-Aware Generation (CAG) for intelligent, context-sensitive responses. Designed for scalable AI applications with pluggable data sources and flexible context handling.

## Features

- Document parsing and processing
- In-memory vector store for embeddings
- Multiple reranking strategies (BM25, LLM-based)
- Interactive query system
- Configurable context handling
- Comprehensive logging

## Components

- `rag_controller.py`: Main controller integrating all components
- `parsing.py`: PDF document parsing and processing
- `embedding.py`: Document embedding and vector store management
- `retrieval.py`: Document retrieval and reranking
- `rerankers.py`: Response reranking strategies

## Setup

1. Clone the repository:
```bash
git clone https://github.com/madelynndinh/rag-cag-system.git
cd rag-cag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the RAG controller:
```bash
python rag_controller.py
```

The system will:
1. Process PDF documents in the `pdf-test` directory
2. Create embeddings and store them in memory
3. Start an interactive query session

## Configuration

The system can be configured through:
- Environment variables
- Command line arguments
- Configuration files

## License

Apache-2.0 