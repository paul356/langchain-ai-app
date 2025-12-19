#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 github.com/paul356
# See LICENSE file for full license text

"""Quick test of vector memory chat functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chat.memory_chat import VectorMemoryChat

def test_vector_chat():
    """Test vector memory chat with a simple session."""
    print("=" * 60)
    print("Testing Vector Memory Chat")
    print("=" * 60)

    # Initialize chat
    print("\n1. Initializing chat...")
    chat = VectorMemoryChat(user_id="test_user")

    # Create new session
    print("\n2. Creating new session...")
    session_id = chat.new_session()
    print(f"   Session ID: {session_id}")

    # Send a few messages
    print("\n3. Testing chat messages...")
    response1 = chat.chat("Hello! My name is Alice.")
    print(f"   User: Hello! My name is Alice.")
    print(f"   AI: {response1[:100]}...")

    response2 = chat.chat("What's the weather like?")
    print(f"   User: What's the weather like?")
    print(f"   AI: {response2[:100]}...")

    response3 = chat.chat("What's my name?")
    print(f"   User: What's my name?")
    print(f"   AI: {response3[:100]}...")

    # Get session info
    print("\n4. Session summary:")
    summary = chat.get_session_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # List sessions
    print("\n5. Listing all sessions...")
    sessions = chat.list_sessions()
    print(f"   Found {len(sessions)} session(s)")
    for sess in sessions:
        print(f"   - {sess['session_id'][:8]}... ({sess['message_count']} messages)")

    # Test resume
    print("\n6. Testing session resume...")
    chat2 = VectorMemoryChat(user_id="test_user")
    success = chat2.resume_session(session_id)
    print(f"   Resume successful: {success}")

    if success:
        response4 = chat2.chat("Can you remind me what we talked about?")
        print(f"   User: Can you remind me what we talked about?")
        print(f"   AI: {response4[:100]}...")

    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_vector_chat()
