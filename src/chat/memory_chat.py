"""
Chat system with vector database memory and session management.
Supports creating new sessions or resuming previous conversations.
"""

import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()


class VectorMemoryChat:
    """Chat system with vector database memory and session management."""

    def __init__(self, user_id: str = "default_user", model_type: str = "ollama"):
        """
        Initialize chat with vector memory.

        Args:
            user_id: Unique identifier for the user
            model_type: Type of LLM to use ("ollama" or "openai")
        """
        self.user_id = user_id
        self.model_type = model_type
        self.session_id = None
        self.current_history = ChatMessageHistory()

        # Setup LLM
        self.llm = self._setup_llm()

        # Setup embeddings (use Ollama for local embeddings)
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # Fast, efficient embedding model
            base_url=ollama_url
        )

        # Setup vector store for persistent memory
        self.vectorstore = Chroma(
            collection_name="chat_history",
            embedding_function=self.embeddings,
            persist_directory="./chroma_chat_db"
        )

        print(f"✅ Vector Memory Chat initialized for user: {user_id}")

    def _setup_llm(self):
        """Setup LLM based on configuration."""
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if self.model_type == "ollama" or ollama_url:
            from langchain_ollama import OllamaLLM
            model = os.getenv("MODEL_NAME", "qwen2.5:latest")
            base_url = ollama_url or "http://localhost:11434"
            print(f"🔧 Using Ollama: {base_url} with model: {model}")
            return OllamaLLM(model=model, base_url=base_url)
        else:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=api_key)

    def new_session(self) -> str:
        """
        Start a new chat session with a unique ID.

        Returns:
            session_id: The unique session identifier
        """
        self.session_id = str(uuid.uuid4())
        self.current_history = ChatMessageHistory()
        print(f"🆕 New session created: {self.session_id}")
        return self.session_id

    def resume_session(self, session_id: str) -> bool:
        """
        Resume a previous chat session.

        Args:
            session_id: The session ID to resume

        Returns:
            bool: True if session found and loaded, False otherwise
        """
        # Query vector store for messages from this session
        try:
            # Use get() with filter to retrieve all matching documents
            results = self.vectorstore.get(
                where={
                    "$and": [
                        {"user_id": {"$eq": self.user_id}},
                        {"session_id": {"$eq": session_id}}
                    ]
                }
            )

            if not results or not results['ids']:
                print(f"❌ No history found for session: {session_id}")
                return False

            # Combine into document-like objects
            documents = []
            for i in range(len(results['ids'])):
                doc = {
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                documents.append(doc)

            # Sort by timestamp
            sorted_docs = sorted(documents, key=lambda x: x['metadata'].get("timestamp", ""))

            # Rebuild chat history
            self.session_id = session_id
            self.current_history = ChatMessageHistory()

            for doc in sorted_docs:
                msg_type = doc['metadata'].get("message_type")
                content = doc['content']

                if msg_type == "human":
                    self.current_history.add_user_message(content)
                elif msg_type == "ai":
                    self.current_history.add_ai_message(content)

            print(f"✅ Resumed session: {session_id} ({len(sorted_docs)} messages loaded)")
            return True

        except Exception as e:
            print(f"❌ Error resuming session: {e}")
            return False

    def list_sessions(self) -> List[Dict]:
        """
        List all sessions for the current user.

        Returns:
            List of session info dictionaries
        """
        try:
            # Get all documents for this user using get() method
            results = self.vectorstore.get(
                where={"user_id": {"$eq": self.user_id}}
            )

            if not results or not results['ids']:
                return []

            # Group by session_id
            sessions = {}
            for i in range(len(results['ids'])):
                metadata = results['metadatas'][i]
                content = results['documents'][i]
                sid = metadata.get("session_id")

                if sid not in sessions:
                    sessions[sid] = {
                        "session_id": sid,
                        "message_count": 0,
                        "first_timestamp": metadata.get("timestamp"),
                        "last_timestamp": metadata.get("timestamp"),
                        "preview": content[:50] + "..." if len(content) > 50 else content
                    }
                sessions[sid]["message_count"] += 1

                # Update timestamps
                current_ts = metadata.get("timestamp")
                if current_ts:
                    if current_ts < sessions[sid]["first_timestamp"]:
                        sessions[sid]["first_timestamp"] = current_ts
                        sessions[sid]["preview"] = content[:50] + "..." if len(content) > 50 else content
                    if current_ts > sessions[sid]["last_timestamp"]:
                        sessions[sid]["last_timestamp"] = current_ts

            # Sort by last timestamp (most recent first)
            session_list = sorted(sessions.values(), key=lambda x: x["last_timestamp"], reverse=True)
            return session_list

        except Exception as e:
            print(f"❌ Error listing sessions: {e}")
            return []

    def _save_to_vector_store(self, message: str, message_type: str):
        """Save a message to the vector store with metadata."""
        if not self.session_id:
            print("⚠️  No active session. Starting new session.")
            self.new_session()

        metadata = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message_type": message_type,
            "timestamp": datetime.now().isoformat()
        }

        self.vectorstore.add_texts(
            texts=[message],
            metadatas=[metadata]
        )

    def _get_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant context from vector store.

        Args:
            query: The current user query
            k: Number of relevant messages to retrieve

        Returns:
            Formatted context string
        """
        if not self.session_id:
            return ""

        # Search for relevant past messages in current session
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter={
                "$and": [
                    {"user_id": {"$eq": self.user_id}},
                    {"session_id": {"$eq": self.session_id}}
                ]
            }
        )

        if not results:
            return ""

        # Format context
        context_parts = []
        for doc in results:
            msg_type = doc.metadata.get("message_type", "unknown")
            context_parts.append(f"[{msg_type}]: {doc.page_content}")

        return "\n".join(context_parts)

    def chat(self, user_input: str, use_context: bool = True) -> str:
        """
        Send a message and get a response.

        Args:
            user_input: The user's message
            use_context: Whether to retrieve relevant context from vector store

        Returns:
            The AI's response
        """
        if not self.session_id:
            print("⚠️  No active session. Starting new session.")
            self.new_session()

        # Add user message to current history
        self.current_history.add_user_message(user_input)
        self._save_to_vector_store(user_input, "human")

        # Build prompt with context
        prompt_parts = []

        # Add relevant context if enabled
        if use_context and len(self.current_history.messages) > 2:
            relevant_context = self._get_relevant_context(user_input, k=3)
            if relevant_context:
                prompt_parts.append(f"Relevant context from earlier:\n{relevant_context}\n")

        # Add recent conversation history (last 5 exchanges)
        recent_messages = self.current_history.messages[-10:]  # Last 5 exchanges (10 messages)
        if recent_messages:
            prompt_parts.append("Recent conversation:")
            for msg in recent_messages[:-1]:  # Exclude the current message
                if isinstance(msg, HumanMessage):
                    prompt_parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    prompt_parts.append(f"Assistant: {msg.content}")

        # Add current query
        prompt_parts.append(f"\nUser: {user_input}")
        prompt_parts.append("Assistant:")

        full_prompt = "\n".join(prompt_parts)

        # Get response from LLM
        response = self.llm.invoke(full_prompt)

        # Add AI response to history
        self.current_history.add_ai_message(response)
        self._save_to_vector_store(response, "ai")

        return response

    def get_session_summary(self) -> Dict:
        """Get summary of the current session."""
        if not self.session_id:
            return {"error": "No active session"}

        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_count": len(self.current_history.messages),
            "active": True
        }

    def clear_current_session(self):
        """Clear the current session from memory (keeps in vector store)."""
        self.current_history = ChatMessageHistory()
        print(f"🗑️  Current session cleared (history preserved in vector store)")


@dataclass
class Command:
    """Represents a chat command with metadata and handler."""
    name: str
    handler: Callable[[Any, str], bool]  # Returns True to continue loop, False to break
    description: str
    usage: Optional[str] = None
    aliases: Optional[List[str]] = None

    def matches(self, input_cmd: str) -> bool:
        """Check if input matches this command or its aliases."""
        cmd_lower = input_cmd.lower()
        if cmd_lower == self.name:
            return True
        if self.aliases:
            return cmd_lower in self.aliases
        return False

    def get_help_text(self) -> str:
        """Generate help text for this command."""
        usage_text = self.usage if self.usage else self.name
        return f"  /{usage_text:<20} - {self.description}"


class CommandRegistry:
    """Registry for managing chat commands."""

    def __init__(self):
        self.commands: Dict[str, Command] = {}

    def register(self, command: Command):
        """Register a command."""
        self.commands[command.name] = command
        # Register aliases
        if command.aliases:
            for alias in command.aliases:
                self.commands[alias] = command

    def get(self, name: str) -> Optional[Command]:
        """Get a command by name or alias."""
        return self.commands.get(name.lower())

    def get_all(self) -> List[Command]:
        """Get all unique commands (excluding aliases)."""
        seen = set()
        unique_commands = []
        for cmd in self.commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                unique_commands.append(cmd)
        return sorted(unique_commands, key=lambda c: c.name)

    def print_help(self):
        """Print help for all commands."""
        print("\n📚 Available Commands:")
        print("=" * 60)
        for cmd in self.get_all():
            print(cmd.get_help_text())
        print("=" * 60)


def create_command_handlers(chat: VectorMemoryChat, state: Dict[str, Any]) -> CommandRegistry:
    """Create and register all command handlers."""
    registry = CommandRegistry()

    # /quit command
    def handle_quit(ctx, args):
        print("👋 Goodbye!")
        return False  # Exit loop

    registry.register(Command(
        name="quit",
        handler=handle_quit,
        description="Exit the chatbot",
        aliases=["exit", "q"]
    ))

    # /help command
    def handle_help(ctx, args):
        registry.print_help()
        return True

    registry.register(Command(
        name="help",
        handler=handle_help,
        description="Show this help message",
        aliases=["h", "?"]
    ))

    # /new command
    def handle_new(ctx, args):
        session_id = ctx.new_session()
        print(f"✅ New session started: {session_id}")
        return True

    registry.register(Command(
        name="new",
        handler=handle_new,
        description="Start a new chat session"
    ))

    # /resume command
    def handle_resume(ctx, args):
        sessions = ctx.list_sessions()
        if not sessions:
            print("❌ No previous sessions found.")
            return True

        print("\n📋 Available sessions:")
        for i, session in enumerate(sessions, 1):
            print(f"{i}. Session: {session['session_id'][:8]}... "
                  f"({session['message_count']} messages, "
                  f"last active: {session['last_timestamp'][:19]})")
            print(f"   Preview: {session['preview']}")

        try:
            idx = int(input(f"\nSelect session (1-{len(sessions)}): ")) - 1
            if 0 <= idx < len(sessions):
                ctx.resume_session(sessions[idx]['session_id'])
            else:
                print("❌ Invalid selection.")
        except ValueError:
            print("❌ Invalid input.")
        return True

    registry.register(Command(
        name="resume",
        handler=handle_resume,
        description="Resume a previous session"
    ))

    # /list command
    def handle_list(ctx, args):
        sessions = ctx.list_sessions()
        if not sessions:
            print("📋 No sessions found.")
        else:
            print(f"\n📋 Found {len(sessions)} session(s):")
            for i, session in enumerate(sessions, 1):
                print(f"\n{i}. Session ID: {session['session_id']}")
                print(f"   Messages: {session['message_count']}")
                print(f"   First message: {session['first_timestamp'][:19]}")
                print(f"   Last message: {session['last_timestamp'][:19]}")
                print(f"   Preview: {session['preview']}")
        return True

    registry.register(Command(
        name="list",
        handler=handle_list,
        description="List all sessions",
        aliases=["ls"]
    ))

    # /info command
    def handle_info(ctx, args):
        summary = ctx.get_session_summary()
        print("\n📊 Session Summary:")
        print("=" * 60)
        for key, value in summary.items():
            print(f"   {key}: {value}")
        print(f"   context_retrieval: {'✅ ON' if state['use_context'] else '❌ OFF'}")
        print("=" * 60)
        return True

    registry.register(Command(
        name="info",
        handler=handle_info,
        description="View current session info",
        aliases=["status"]
    ))

    # /clear command
    def handle_clear(ctx, args):
        ctx.clear_current_session()
        return True

    registry.register(Command(
        name="clear",
        handler=handle_clear,
        description="Clear current session memory"
    ))

    # /context command
    def handle_context(ctx, args):
        parts = args.split()
        if len(parts) < 1:
            current = 'ON' if state['use_context'] else 'OFF'
            print(f"⚠️  Usage: /context on|off (currently: {current})")
            return True

        if parts[0].lower() == 'on':
            state['use_context'] = True
            print("✅ Context retrieval enabled")
        elif parts[0].lower() == 'off':
            state['use_context'] = False
            print("❌ Context retrieval disabled")
        else:
            print("⚠️  Usage: /context on|off")
        return True

    registry.register(Command(
        name="context",
        handler=handle_context,
        description="Toggle context retrieval",
        usage="context on|off"
    ))

    return registry


def main():
    """Demo the vector memory chat system."""
    print("=" * 60)
    print("Vector Memory Chat System")
    print("=" * 60)
    print("Type /help for available commands")
    print("-" * 60)

    # Initialize chat
    chat = VectorMemoryChat(user_id="alice")

    # Shared state accessible by command handlers
    state = {"use_context": True}

    # Create command registry
    commands = create_command_handlers(chat, state)

    # Start with a new session automatically
    chat.new_session()

    # Single-level chat loop with commands
    while True:
        # Flush any incomplete input before prompting
        import sys
        if sys.stdin.isatty():
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)

        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith('/'):
            # Parse command and arguments
            parts = user_input[1:].split(maxsplit=1)

            # Handle empty command (just "/")
            if not parts or not parts[0]:
                print("⚠️  Empty command.")
                commands.print_help()
                continue

            cmd_name = parts[0]
            cmd_args = parts[1] if len(parts) > 1 else ""

            # Find and execute command
            command = commands.get(cmd_name)
            if command:
                should_continue = command.handler(chat, cmd_args)
                if not should_continue:
                    break
            else:
                print(f"⚠️  Unrecognized command: /{cmd_name}")
                commands.print_help()
            continue

        # Regular chat
        if not chat.session_id:
            print("⚠️  No active session. Starting new session automatically.")
            chat.new_session()

        print("🤖 Assistant: ", end="", flush=True)
        response = chat.chat(user_input, use_context=state['use_context'])
        print(response)


if __name__ == "__main__":
    main()
