# LangChain AI Application

A gradually increasing complexity AI application built with LangChain, starting from basic chat to advanced agents.

## Features Roadmap

### ✅ Phase 1: Basic Setup
- [x] Environment setup
- [x] LangChain installation
- [x] Simple LLM chat

### ✅ Phase 2: Chat Application
- [x] Basic chat with OpenAI/Local LLM
- [x] Prompt templates
- [x] Conversation flow

### ✅ Phase 3: Memory & Context
- [x] Conversation memory
- [x] Vector database memory (ChromaDB)
- [x] Session management
- [x] Context retrieval from chat history

### ✅ Phase 4: Document Q&A (RAG)
- [x] Document processing (.txt, .pdf)
- [x] Vector embeddings (Ollama)
- [x] Knowledge base storage
- [x] Retrieval system
- [x] Context-aware question answering

### ✅ Phase 5: Web Interface
- [x] FastAPI backend
- [x] REST API endpoints
- [x] Document upload
- [x] Session management API
- [x] Knowledge base API

### 🤖 Phase 6: Agents & Tools
- [ ] LangChain agents
- [ ] Tool integration
- [ ] Autonomous task solving

### 🚀 Phase 7: Production Features
- [ ] Streaming responses
- [ ] Error handling
- [ ] Logging
- [ ] Deployment

## Project Structure

```
langchain-ai-app/
├── src/
│   ├── __init__.py
│   ├── chat/               # Chat functionality
│   ├── rag/                # Document Q&A
│   ├── agents/             # Agent implementations
│   └── api/                # FastAPI endpoints
├── data/                   # Sample documents
├── notebooks/              # Jupyter examples
├── tests/
├── requirements.txt
├── .env.template
└── README.md
```

## Getting Started

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.template .env
# Edit .env with your API keys
```

4. Run the application:
```bash
python src/main.py
```

## Model Configuration

This application supports multiple LLM providers through a unified model factory:

### Supported Providers

1. **OpenAI** - GPT-3.5, GPT-4, and other OpenAI models
2. **Ollama** - Local models (Llama2, Mistral, Qwen3, etc.)
3. **Qwen** - Alibaba Cloud's Qwen models via Dashscope API

### Configuration

Configure your model by setting environment variables in `.env`:

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your preferred model provider
```

#### Option 1: OpenAI Models

```bash
MODEL_PROVIDER=openai
MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=your_openai_api_key
```

Get your API key from: https://platform.openai.com/api-keys

#### Option 2: Local Ollama Models

```bash
MODEL_PROVIDER=ollama
MODEL_NAME=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434
```

Setup Ollama:
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull qwen3:latest`
3. Start Ollama service (usually runs automatically)

#### Option 3: Qwen Cloud API (Recommended for Qwen3-max)

```bash
MODEL_PROVIDER=qwen
MODEL_NAME=qwen-max
QWEN_API_KEY=your_dashscope_api_key
```

Get your API key from: https://dashscope.console.aliyun.com/

**Available Qwen Models:**
- `qwen-max` - Latest and most capable model
- `qwen3-max` - Latest Qwen 3 series model
- `qwen-plus` - Balanced performance and cost
- `qwen-turbo` - Fast and cost-effective
- `qwen-long` - Extended context window (10M tokens)

### Embedding Model Configuration

Configure embeddings for vector database and RAG:

```bash
# Choose embedding provider
EMBEDDING_PROVIDER=ollama  # Options: openai, ollama, dashscope

# Set embedding model (provider-specific)
# - OpenAI: text-embedding-3-small, text-embedding-3-large
# - Ollama: nomic-embed-text, mxbai-embed-large
# - Dashscope: text-embedding-v1, text-embedding-v2, text-embedding-v3
EMBEDDING_MODEL=nomic-embed-text
```

**Recommended Configurations:**
- **Local (Free)**: `EMBEDDING_PROVIDER=ollama` with `EMBEDDING_MODEL=nomic-embed-text`
- **Cloud (OpenAI)**: `EMBEDDING_PROVIDER=openai` with `EMBEDDING_MODEL=text-embedding-3-small`
- **Cloud (Dashscope)**: `EMBEDDING_PROVIDER=dashscope` with `EMBEDDING_MODEL=text-embedding-v3` or `text-embedding-v4`

### Additional Configuration

```bash
# Adjust model temperature (0.0-1.0)
MODEL_TEMPERATURE=0.7
```

### Quick Start

```bash
# Interactive setup wizard
python test_model_config.py --setup

# Or test existing configuration
python test_model_config.py
```

### Documentation

- 📖 **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 2 minutes
- 📚 **[Complete Model Configuration Guide](docs/MODEL_CONFIGURATION.md)** - Detailed setup and troubleshooting
- 🔢 **[Embedding Configuration Guide](docs/EMBEDDING_CONFIGURATION.md)** - Configure embeddings for vector databases and RAG

## Usage Examples

Each phase will include practical examples and code demonstrations.