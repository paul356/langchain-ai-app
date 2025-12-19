# MIT License
# Copyright (c) 2025 github.com/paul356
# See LICENSE file for full license text

"""
Test client for Vector Memory Chat API.
Demonstrates how to interact with the FastAPI endpoints.
"""

import requests
import json
from typing import Optional

# API base URL
BASE_URL = "http://localhost:8000"


class ChatAPIClient:
    """Client for interacting with the Vector Memory Chat API."""

    def __init__(self, base_url: str = BASE_URL, user_id: str = "default_user"):
        self.base_url = base_url
        self.user_id = user_id
        self.session_id: Optional[str] = None

    def check_status(self):
        """Check API status."""
        response = requests.get(f"{self.base_url}/")
        return response.json()

    def new_session(self):
        """Create a new chat session."""
        response = requests.post(
            f"{self.base_url}/session/new",
            json={"user_id": self.user_id}
        )
        data = response.json()
        self.session_id = data.get("session_id")
        return data

    def resume_session(self, session_id: str):
        """Resume a previous session."""
        response = requests.post(
            f"{self.base_url}/session/resume",
            json={"user_id": self.user_id, "session_id": session_id}
        )
        data = response.json()
        self.session_id = session_id
        return data

    def list_sessions(self):
        """List all sessions for the user."""
        response = requests.get(
            f"{self.base_url}/session/list",
            params={"user_id": self.user_id}
        )
        return response.json()

    def get_session_info(self):
        """Get current session info."""
        response = requests.get(
            f"{self.base_url}/session/info",
            params={"user_id": self.user_id}
        )
        return response.json()

    def clear_session(self):
        """Clear current session."""
        response = requests.post(
            f"{self.base_url}/session/clear",
            json={"user_id": self.user_id}
        )
        return response.json()

    def chat(self, message: str, use_context: bool = True, use_knowledge: bool = True):
        """Send a chat message."""
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "message": message,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "use_context": use_context,
                "use_knowledge": use_knowledge
            }
        )
        data = response.json()

        # Update session_id from response
        if "session_id" in data:
            self.session_id = data["session_id"]

        return data

    def upload_document(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Upload a document to the knowledge base."""
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {
                "user_id": self.user_id,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            response = requests.post(
                f"{self.base_url}/knowledge/upload",
                files=files,
                data=data
            )
        return response.json()

    def upload_folder(self, folder_path: str = "./data"):
        """Upload all documents from a folder."""
        response = requests.post(
            f"{self.base_url}/knowledge/upload-folder",
            data={"user_id": self.user_id, "folder_path": folder_path}
        )
        return response.json()

    def search_knowledge(self, query: str, k: int = 3):
        """Search the knowledge base."""
        response = requests.get(
            f"{self.base_url}/knowledge/search",
            params={"query": query, "user_id": self.user_id, "k": k}
        )
        return response.json()


def main():
    """Demo the API client."""
    print("=" * 60)
    print("Vector Memory Chat API - Test Client")
    print("=" * 60)

    # Initialize client
    client = ChatAPIClient(user_id="test_user")

    # Check API status
    print("\n1. Checking API status...")
    status = client.check_status()
    print(f"   Status: {status}")

    # Create new session
    print("\n2. Creating new session...")
    session = client.new_session()
    print(f"   Session ID: {session['session_id']}")

    # Send some messages
    print("\n3. Chatting...")
    messages = [
        "Hello! What can you help me with?",
        "Tell me about LangChain.",
        "What is a vector database?"
    ]

    for msg in messages:
        print(f"\n   User: {msg}")
        response = client.chat(msg)
        print(f"   Assistant: {response['response'][:100]}...")

    # Get session info
    print("\n4. Getting session info...")
    info = client.get_session_info()
    print(f"   Session info: {json.dumps(info, indent=2)}")

    # List all sessions
    print("\n5. Listing all sessions...")
    sessions = client.list_sessions()
    print(f"   Found {sessions['count']} session(s)")

    # Upload a document (if exists)
    print("\n6. Testing knowledge base upload...")
    try:
        result = client.upload_folder("./data")
        print(f"   Upload result: {result}")
    except Exception as e:
        print(f"   Upload skipped: {e}")

    # Search knowledge base
    print("\n7. Searching knowledge base...")
    search_result = client.search_knowledge("What is LangChain?")
    print(f"   Search results: {search_result}")

    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    main()
