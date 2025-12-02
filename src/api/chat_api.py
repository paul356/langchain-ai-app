"""
FastAPI backend for Vector Memory Chat system.
Provides REST API endpoints for chat, session management, and knowledge base operations.
"""

import os
import sys
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.memory_chat import VectorMemoryChat


# Pydantic models for request/response
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's message")
    user_id: str = Field(default="default_user", description="User identifier")
    session_id: Optional[str] = Field(None, description="Session ID (optional, for resuming)")
    use_context: bool = Field(default=True, description="Enable context retrieval")
    use_knowledge: bool = Field(default=True, description="Enable knowledge base retrieval")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI assistant's response")
    session_id: str = Field(..., description="Current session ID")
    timestamp: str = Field(..., description="Response timestamp")


class SessionRequest(BaseModel):
    """Request model for session operations."""
    user_id: str = Field(default="default_user", description="User identifier")
    session_id: Optional[str] = Field(None, description="Session ID (for resume)")


class SessionInfo(BaseModel):
    """Session information model."""
    session_id: str
    message_count: int
    first_timestamp: str
    last_timestamp: str
    preview: str


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""
    sessions: List[SessionInfo]
    count: int


class UploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    message: str
    chunks_indexed: Optional[int] = None


class StatusResponse(BaseModel):
    """Response model for status check."""
    status: str
    message: str


# FastAPI application
app = FastAPI(
    title="Vector Memory Chat API",
    description="REST API for chat system with vector database memory and knowledge base",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active chat instances (keyed by user_id)
active_chats: Dict[str, VectorMemoryChat] = {}

# Get the path to the frontend directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Mount static files for the frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def get_or_create_chat(user_id: str) -> VectorMemoryChat:
    """Get existing chat instance or create a new one for the user."""
    if user_id not in active_chats:
        active_chats[user_id] = VectorMemoryChat(user_id=user_id)
    return active_chats[user_id]


@app.get("/")
async def root():
    """Root endpoint - Serve the frontend."""
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    return RedirectResponse(url="/docs")


@app.get("/api/status", response_model=StatusResponse)
async def api_status():
    """API status check endpoint."""
    return StatusResponse(
        status="online",
        message="Vector Memory Chat API is running"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a response.

    - **message**: The user's message
    - **user_id**: User identifier (default: "default_user")
    - **session_id**: Optional session ID to resume a conversation
    - **use_context**: Enable/disable context retrieval (default: true)
    - **use_knowledge**: Enable/disable knowledge base retrieval (default: true)
    """
    try:
        chat_instance = get_or_create_chat(request.user_id)

        # Resume session if provided, otherwise start new if none active
        if request.session_id:
            if not chat_instance.session_id or chat_instance.session_id != request.session_id:
                success = chat_instance.resume_session(request.session_id)
                if not success:
                    raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
        elif not chat_instance.session_id:
            chat_instance.new_session()

        # Get response
        response = chat_instance.chat(
            request.message,
            use_context=request.use_context,
            use_knowledge=request.use_knowledge
        )

        return ChatResponse(
            response=response,
            session_id=chat_instance.session_id,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.post("/session/new")
async def create_session(request: SessionRequest):
    """
    Create a new chat session.

    - **user_id**: User identifier
    """
    try:
        chat_instance = get_or_create_chat(request.user_id)
        session_id = chat_instance.new_session()

        return {
            "session_id": session_id,
            "user_id": request.user_id,
            "message": "New session created"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


@app.post("/session/resume")
async def resume_session(request: SessionRequest):
    """
    Resume a previous chat session.

    - **user_id**: User identifier
    - **session_id**: Session ID to resume
    """
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        chat_instance = get_or_create_chat(request.user_id)
        success = chat_instance.resume_session(request.session_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

        summary = chat_instance.get_session_summary()
        return {
            "message": "Session resumed successfully",
            "session_info": summary
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming session: {str(e)}")


@app.get("/session/list", response_model=SessionListResponse)
async def list_sessions(user_id: str = "default_user"):
    """
    List all sessions for a user.

    - **user_id**: User identifier
    """
    try:
        chat_instance = get_or_create_chat(user_id)
        sessions = chat_instance.list_sessions()

        return SessionListResponse(
            sessions=[SessionInfo(**session) for session in sessions],
            count=len(sessions)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")


@app.get("/session/info")
async def get_session_info(user_id: str = "default_user"):
    """
    Get information about the current active session.

    - **user_id**: User identifier
    """
    try:
        chat_instance = get_or_create_chat(user_id)
        summary = chat_instance.get_session_summary()

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session info: {str(e)}")


@app.get("/session/history")
async def get_session_history(user_id: str = "default_user", session_id: str = None):
    """
    Get the full message history for a session.

    - **user_id**: User identifier
    - **session_id**: Session ID to retrieve history for
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        chat_instance = get_or_create_chat(user_id)
        history = chat_instance.get_session_history(session_id)

        return {
            "session_id": session_id,
            "user_id": user_id,
            "history": history,
            "count": len(history)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session history: {str(e)}")


@app.post("/session/clear")
async def clear_session(request: SessionRequest):
    """
    Clear the current session from memory (preserves in vector store).

    - **user_id**: User identifier
    """
    try:
        chat_instance = get_or_create_chat(request.user_id)
        chat_instance.clear_current_session()

        return {
            "message": "Session cleared successfully",
            "user_id": request.user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@app.post("/knowledge/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=100)
):
    """
    Upload a document to the knowledge base.

    - **file**: Document file (.txt or .pdf)
    - **user_id**: User identifier
    - **chunk_size**: Size of text chunks for indexing
    - **chunk_overlap**: Overlap between chunks
    """
    # Validate file type
    allowed_extensions = [".txt", ".pdf"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file temporarily
    temp_file = None
    try:
        chat_instance = get_or_create_chat(user_id)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Upload to knowledge base
        success = chat_instance.upload_document(
            temp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if success:
            return UploadResponse(
                success=True,
                message=f"Document '{file.filename}' uploaded successfully"
            )
        else:
            return UploadResponse(
                success=False,
                message=f"Failed to upload document '{file.filename}'"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)
        file.file.close()


@app.post("/knowledge/upload-folder")
async def upload_folder(
    user_id: str = Form(default="default_user"),
    folder_path: str = Form(default="./data")
):
    """
    Upload all documents from a folder to the knowledge base.

    - **user_id**: User identifier
    - **folder_path**: Path to folder containing documents
    """
    try:
        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")

        chat_instance = get_or_create_chat(user_id)
        count = chat_instance.upload_documents_from_folder(folder_path)

        return {
            "success": True,
            "message": f"Uploaded {count} documents from {folder_path}",
            "documents_count": count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading folder: {str(e)}")


@app.get("/knowledge/search")
async def search_knowledge(
    query: str,
    user_id: str = "default_user",
    k: int = 3
):
    """
    Search the knowledge base for relevant information.

    - **query**: Search query
    - **user_id**: User identifier
    - **k**: Number of results to return
    """
    try:
        chat_instance = get_or_create_chat(user_id)
        results = chat_instance.retrieve_knowledge(query, k=k)

        return {
            "query": query,
            "results": results,
            "count": k
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching knowledge base: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Vector Memory Chat API")
    print("=" * 60)
    print("Starting FastAPI server...")
    print("API documentation: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
