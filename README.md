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

## API Keys Required

- **OpenAI API Key**: For GPT models (get from https://platform.openai.com/)
- **LangSmith API Key**: For tracing (optional, get from https://smith.langchain.com/)

## Alternative: Local Models

You can also use local models with Ollama instead of OpenAI:

1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Update .env to use local model

## Usage Examples

Each phase will include practical examples and code demonstrations.