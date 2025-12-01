# FastAPI Backend for Vector Memory Chat

## Overview
A REST API built with FastAPI that provides endpoints for the Vector Memory Chat system.

## Starting the Server

```bash
# From the project root
python src/api/chat_api.py
```

Or with uvicorn directly:
```bash
uvicorn src.api.chat_api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Interactive docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative docs (ReDoc)**: http://localhost:8000/redoc

## Endpoints

### Status
- `GET /` - Check API status

### Chat
- `POST /chat` - Send a message and get a response
  ```json
  {
    "message": "Hello!",
    "user_id": "user123",
    "session_id": "optional-session-id",
    "use_context": true,
    "use_knowledge": true
  }
  ```

### Session Management
- `POST /session/new` - Create a new chat session
- `POST /session/resume` - Resume a previous session
- `GET /session/list?user_id=user123` - List all sessions
- `GET /session/info?user_id=user123` - Get current session info
- `POST /session/clear` - Clear current session

### Knowledge Base
- `POST /knowledge/upload` - Upload a document (multipart/form-data)
- `POST /knowledge/upload-folder` - Upload all documents from a folder
- `GET /knowledge/search?query=...&user_id=...&k=3` - Search knowledge base

## Usage Examples

### cURL Examples

**Chat:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "user_id": "user123"}'
```

**Create Session:**
```bash
curl -X POST "http://localhost:8000/session/new" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'
```

**Upload Document:**
```bash
curl -X POST "http://localhost:8000/knowledge/upload" \
  -F "file=@/path/to/document.pdf" \
  -F "user_id=user123"
```

**List Sessions:**
```bash
curl "http://localhost:8000/session/list?user_id=user123"
```

### Python Client

Use the provided test client:
```bash
python src/api/test_client.py
```

Or import the client in your code:
```python
from src.api.test_client import ChatAPIClient

client = ChatAPIClient(user_id="my_user")
client.new_session()
response = client.chat("Hello!")
print(response['response'])
```

## Architecture

The API maintains in-memory `VectorMemoryChat` instances per user. Each user's chat state is preserved across requests through the user_id parameter.

- **Vector Store**: ChromaDB (persisted to disk)
- **Collections**:
  - `chat_history` - Stores conversation messages
  - `knowledge_base` - Stores uploaded documents
- **LLM**: Configurable (Ollama or OpenAI)

## Features

✅ RESTful API with automatic documentation
✅ Session management (create, resume, list)
✅ Context-aware responses using vector search
✅ Knowledge base integration
✅ Document upload (.txt, .pdf)
✅ CORS support for frontend integration
✅ Pydantic validation
✅ Error handling

## Configuration

Configure via environment variables (`.env` file):
```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=qwen2.5:latest
```

## Next Steps

- Add authentication/authorization
- Implement rate limiting
- Add streaming responses (SSE)
- Add websocket support for real-time chat
- Deploy with Docker
