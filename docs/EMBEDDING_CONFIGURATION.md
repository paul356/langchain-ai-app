# Embedding Model Configuration Guide

This guide explains how to configure embedding models for vector databases and RAG (Retrieval-Augmented Generation) in the LangChain AI Application.

## Overview

Embeddings are used to convert text into numerical vectors for:
- **Vector Database Storage**: Storing chat history and documents
- **Semantic Search**: Finding relevant context from past conversations
- **Knowledge Base RAG**: Retrieving relevant documents for question answering

## Supported Providers

### 1. Ollama (Local, Free)

**Best for**: Local development, privacy-conscious deployments, no API costs

```bash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

**Available Models**:
- `nomic-embed-text` - Fast, efficient, recommended for most use cases (768 dimensions)
- `mxbai-embed-large` - Larger model for better accuracy (1024 dimensions)
- `all-minilm` - Lightweight, faster inference (384 dimensions)

**Setup**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text
```

### 2. OpenAI (Cloud)

**Best for**: High accuracy, production deployments with OpenAI already in use

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-your-api-key-here
```

**Available Models**:
- `text-embedding-3-small` - Cost-effective, good performance (1536 dimensions)
- `text-embedding-3-large` - Higher accuracy, more expensive (3072 dimensions)
- `text-embedding-ada-002` - Legacy model, still supported (1536 dimensions)

**Pricing** (as of 2024):
- `text-embedding-3-small`: $0.02 per 1M tokens
- `text-embedding-3-large`: $0.13 per 1M tokens

### 3. Dashscope (Alibaba Cloud)

**Best for**: China deployments, Qwen ecosystem integration

```bash
EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=text-embedding-v3
DASHSCOPE_API_KEY=sk-your-dashscope-key
```

**Available Models**:
- `text-embedding-v1` - Basic embedding model
- `text-embedding-v2` - Improved performance
- `text-embedding-v3` - Current recommended model
- `text-embedding-v4` - Latest model with enhanced capabilities (if available)

**Get API Key**: https://dashscope.console.aliyun.com/

## Configuration Examples

### Example 1: Local Development (Free)

```bash
# .env file
MODEL_PROVIDER=ollama
MODEL_NAME=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```

### Example 2: Production with OpenAI

```bash
# .env file
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4
OPENAI_API_KEY=sk-...

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
# Uses same OPENAI_API_KEY
```

### Example 3: Qwen Cloud + Dashscope Embeddings

```bash
# .env file
MODEL_PROVIDER=qwen
MODEL_NAME=qwen-max
QWEN_API_KEY=sk-...

EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=text-embedding-v3
DASHSCOPE_API_KEY=sk-...
# Or uses same QWEN_API_KEY if not specified
```

### Example 4: Mixed Providers

```bash
# .env file
# Use local Ollama for LLM
MODEL_PROVIDER=ollama
MODEL_NAME=qwen3:latest

# Use cloud Dashscope for embeddings
EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=sk-...
```

## Usage in Code

### Using the Model Factory

```python
from models import get_llm, get_embeddings, ModelConfig

# Load configuration from environment
config = ModelConfig.from_env()

# Get LLM
llm = get_llm(config)

# Get embeddings
embeddings = get_embeddings(config)

# Use with ChromaDB
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

### Custom Configuration

```python
from models import ModelConfig, get_embeddings

# Create custom config
config = ModelConfig(
    provider="qwen",
    model_name="qwen-max",
    embedding_provider="dashscope",
    embedding_model="text-embedding-v4",
    embedding_api_key="your-key-here"
)

embeddings = get_embeddings(config)
```

## Performance Considerations

### Model Dimensions

- **Higher dimensions** (e.g., 3072): Better accuracy, more storage, slower search
- **Lower dimensions** (e.g., 384): Faster search, less storage, slightly lower accuracy

### Best Practices

1. **Development**: Use Ollama with `nomic-embed-text` for fast, free testing
2. **Production**:
   - Use OpenAI `text-embedding-3-small` for balanced performance/cost
   - Use Dashscope `text-embedding-v3` for China deployments
3. **Consistency**: Always use the same embedding model for a vector database
4. **Migration**: If changing models, re-index all documents

## Troubleshooting

### Error: "DASHSCOPE_API_KEY not found"

**Solution**: Set either `DASHSCOPE_API_KEY` or `QWEN_API_KEY` in your `.env` file:
```bash
DASHSCOPE_API_KEY=sk-your-key-here
```

### Error: "Ollama connection refused"

**Solution**: Ensure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (if not running)
ollama serve
```

### Error: "Model not found"

**Solution**: Pull the embedding model first:
```bash
ollama pull nomic-embed-text
```

### Mixing Embedding Models

**Problem**: Using different embedding models for the same vector database causes search issues.

**Solution**:
1. Stick with one embedding model per database
2. If you must change models, delete and re-create the vector database:
```bash
rm -rf ./chroma_chat_db
# Then re-upload documents
```

## Testing Your Configuration

```bash
# Test model configuration including embeddings
python src/models/model_factory.py

# Expected output:
# ============================================================
# 📋 Current Model Configuration
# ============================================================
# Provider:    ollama
# Model:       qwen3:latest
# Temperature: 0.7
# Base URL:    http://localhost:11434
#
# ------------------------------------------------------------
# 📊 Embedding Configuration
# ------------------------------------------------------------
# Provider:    ollama
# Model:       nomic-embed-text
# ============================================================
#
# ✅ LLM loaded successfully!
# ✅ Embeddings loaded successfully!
```

## API References

- **LangChain Embeddings**: https://python.langchain.com/docs/modules/data_connection/text_embedding/
- **Ollama Embeddings**: https://ollama.ai/blog/embedding-models
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
- **Dashscope API**: https://help.aliyun.com/zh/dashscope/
