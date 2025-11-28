"""
Advanced prompt templates and techniques for LangChain.
Demonstrates different prompting strategies.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

load_dotenv()


class PersonalityResponse(BaseModel):
    """Structured response with personality analysis."""
    response: str = Field(description="The main response to the user")
    tone: str = Field(description="The tone of the response (e.g., friendly, professional, humorous)")
    confidence: float = Field(description="Confidence level from 0.0 to 1.0")


class AdvancedChatBot:
    """Chatbot with advanced prompting techniques."""

    def __init__(self, model_type: str = "ollama"):
        """Initialize with different prompting strategies."""
        self.model_type = model_type
        self.llm = self._setup_llm()
        self.chat_history = []
        # Feature flags - multiple can be active at once
        self.features = {
            "role": "helpful_assistant",  # Current role persona
            "cot": False,  # Chain of thought reasoning
            "context": None,  # User context metadata
        }
        # Conversation flow settings
        self.flow_config = {
            "max_history": 10,  # Maximum exchanges to keep in history
            "window_size": 5,   # Number of recent exchanges to include in context
            "detect_topics": False,  # Enable topic change detection
            "summarize_old": False,  # Summarize old history instead of truncating
        }
        self.conversation_summary = None  # Stores summary of older conversations

    def _setup_llm(self):
        """Setup LLM - prefer Ollama if configured."""
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if self.model_type == "ollama" or ollama_url:
            from langchain_ollama import OllamaLLM
            model = os.getenv("MODEL_NAME", "qwen3:latest")
            base_url = ollama_url or "http://localhost:11434"
            print(f"🔧 Advanced prompts using Ollama: {base_url} with model: {model}")
            return OllamaLLM(model=model, base_url=base_url)
        else:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=api_key)

    def _format_history(self):
        """Format chat history into a list of messages with flow management."""
        messages = []

        # Get relevant history based on flow config
        window_size = self.flow_config["window_size"]
        relevant_history = self.chat_history[-window_size:] if window_size > 0 else self.chat_history

        # Add summary of older conversations if enabled and exists
        if self.conversation_summary and self.flow_config["summarize_old"]:
            messages.append(AIMessage(content=f"[Previous conversation summary: {self.conversation_summary}]"))

        # Add recent history
        for exchange in relevant_history:
            messages.append(HumanMessage(content=exchange["user"]))
            messages.append(AIMessage(content=exchange["bot"]))

        return messages

    def set_role(self, role: str):
        """Set the chatbot's personality role."""
        self.features["role"] = role
        print(f"🔄 Role set to: {role}")

    def enable_cot(self):
        """Enable Chain of Thought reasoning."""
        self.features["cot"] = True
        print("🔄 Chain of Thought enabled.")

    def disable_cot(self):
        """Disable Chain of Thought reasoning."""
        self.features["cot"] = False
        print("🔄 Chain of Thought disabled.")

    def set_context(self, context: Dict[str, Any]):
        """Set user context metadata."""
        self.features["context"] = context
        print(f"🔄 Context set to: {context}")

    def clear_context(self):
        """Clear user context."""
        self.features["context"] = None
        print("🔄 Context cleared.")

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        self.conversation_summary = None
        print("🧹 History cleared.")

    def configure_flow(self, max_history: int = None, window_size: int = None,
                      detect_topics: bool = None, summarize_old: bool = None):
        """Configure conversation flow settings."""
        if max_history is not None:
            self.flow_config["max_history"] = max_history
        if window_size is not None:
            self.flow_config["window_size"] = window_size
        if detect_topics is not None:
            self.flow_config["detect_topics"] = detect_topics
        if summarize_old is not None:
            self.flow_config["summarize_old"] = summarize_old
        print(f"🔄 Flow config updated: {self.flow_config}")

    def _manage_history_length(self):
        """Manage history length based on flow config."""
        max_history = self.flow_config["max_history"]
        window_size = self.flow_config["window_size"]

        if len(self.chat_history) > max_history:
            # If summarization is enabled, summarize old messages and truncate to window_size
            if self.flow_config["summarize_old"]:
                keep_size = window_size if window_size > 0 else max_history
                # Summarize everything except the last window_size exchanges
                old_exchanges = self.chat_history[:-keep_size]
                self._summarize_history(old_exchanges)
                # Truncate to window_size
                self.chat_history = self.chat_history[-keep_size:]
                print(f"🔄 History summarized and trimmed to last {keep_size} exchanges")
            else:
                # Just truncate to max_history without summarizing
                self.chat_history = self.chat_history[-max_history:]
                print(f"🔄 History trimmed to last {max_history} exchanges")

    def _summarize_history(self, exchanges: List[Dict[str, str]]):
        """Summarize old conversation history."""
        if not exchanges:
            return

        import re

        # Build conversation text, removing <think> tags from bot responses
        conv_parts = []

        # Include previous summary if exists
        if self.conversation_summary:
            conv_parts.append(f"Previous summary: {self.conversation_summary}\n")

        # Add new exchanges
        for ex in exchanges:
            user_msg = ex['user']
            # Clean bot response of <think> tags
            bot_msg = re.sub(r'<think>.*?</think>', '', ex['bot'], flags=re.DOTALL).strip()
            conv_parts.append(f"User: {user_msg}\nAssistant: {bot_msg}")

        conv_text = "\n".join(conv_parts)

        # Create summarization prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the entire conversation on both sides (including any previous summary) in concise sentences."),
            ("human", "Conversation:\n{conversation}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            new_summary = chain.invoke({"conversation": conv_text})

            # Clean up the summary - remove <think> tags if still present
            self.conversation_summary = re.sub(r'<think>.*?</think>', '', new_summary, flags=re.DOTALL).strip()

            print("📝 Previous conversation summarized")
        except Exception as e:
            print(f"⚠️  Failed to summarize history: {e}")

    def _detect_topic_change(self, user_input: str) -> bool:
        """Detect if the user is changing topics."""
        if not self.chat_history or not self.flow_config["detect_topics"]:
            return False

        # Get last few exchanges
        recent_messages = self.chat_history[-3:]
        recent_text = " ".join([ex["user"] for ex in recent_messages])

        # Build topic detection prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze if the new message represents a topic change from the previous conversation. "
                      "Answer with only 'YES' or 'NO'."),
            ("human", "Previous conversation: {previous}\n\nNew message: {current}\n\nIs this a topic change?")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"previous": recent_text, "current": user_input})
            return "YES" in result.upper()
        except Exception:
            return False

    def _get_conversation_state(self) -> str:
        """Get current conversation state."""
        history_len = len(self.chat_history)

        if history_len == 0:
            return "greeting"
        elif history_len < 3:
            return "opening"
        else:
            return "ongoing"

    def chat(self, user_input: str) -> str:
        """Main chat entry point that builds prompt from active features."""
        try:
            # Validate input
            if not user_input.strip():
                return "Please provide a message."

            # Detect topic change if enabled
            if self._detect_topic_change(user_input):
                print("🔄 Topic change detected")

            # Get conversation state
            state = self._get_conversation_state()

            # Build and invoke with state awareness
            response = self._build_and_invoke(user_input, state)

            # Add to history
            self.chat_history.append({"user": user_input, "bot": response})

            # Manage history length
            self._manage_history_length()

            return response
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def _build_and_invoke(self, user_input: str, state: str = "ongoing") -> str:
        """Build prompt dynamically based on active features and conversation state."""
        # Build system prompt by combining active features
        system_parts = []

        # Add role personality
        role = self.features["role"]
        role_prompts = {
            "helpful_assistant": "You are a helpful and knowledgeable assistant. Provide clear, accurate, and friendly responses.",
            "creative_writer": "You are a creative writer and storyteller. Respond with imagination, vivid descriptions, and engaging narratives.",
            "technical_expert": "You are a technical expert and engineer. Provide precise, detailed, and technically accurate responses with examples.",
            "philosopher": "You are a wise philosopher. Respond with deep thoughts, ask meaningful questions, and explore ideas thoroughly.",
            "comedian": "You are a friendly comedian. Respond with humor, wit, and light-heartedness while still being helpful."
        }
        system_parts.append(role_prompts.get(role, role_prompts["helpful_assistant"]))

        # Add conversation state guidance
        if state == "greeting":
            system_parts.append("\n\nThis is the start of a new conversation. Be welcoming and establish rapport.")
        elif state == "opening":
            system_parts.append("\n\nYou are in the early stages of conversation. Build on previous exchanges naturally.")

        # Add CoT instructions if enabled
        if self.features["cot"]:
            system_parts.append(
                "\n\nWhen answering, think through problems step by step. "
                "Format your response as:\n"
                "**Problem Analysis:** [Break down the problem]\n"
                "**Step-by-step Solution:** [Numbered steps with reasoning]\n"
                "**Final Answer:** [Your conclusion]"
            )

        # Add context if set
        context_dict = {}
        if self.features["context"]:
            context = self.features["context"]
            context_dict = {
                "user_name": context.get("user_name", "User"),
                "location": context.get("location", "Unknown"),
                "interests": context.get("interests", "Various topics")
            }
            system_parts.append(
                f"\n\nContext Information:\n"
                f"- User Name: {{user_name}}\n"
                f"- Location: {{location}}\n"
                f"- Interests: {{interests}}\n"
                f"Use this context naturally in your response when relevant."
            )

        # Combine system message
        system_message = "".join(system_parts)

        # Build prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Build input dict
        input_dict = {"input": user_input}
        input_dict.update(context_dict)

        # Create chain
        chain = (
            {
                **{k: lambda x, key=k: input_dict.get(key, x.get(key) if isinstance(x, dict) else x) for k in input_dict.keys()},
                "history": lambda x: self._format_history()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(input_dict)

    def few_shot_learning(self, user_input: str, task_type: str = "classification") -> str:
        """Demonstrate few-shot learning with examples."""

        if task_type == "classification":
            # Few-shot examples for sentiment classification
            examples = [
                {"input": "I love this product! It's amazing!", "output": "Positive"},
                {"input": "This is terrible, I hate it.", "output": "Negative"},
                {"input": "It's okay, nothing special.", "output": "Neutral"},
                {"input": "Best purchase ever! Highly recommend!", "output": "Positive"},
                {"input": "Waste of money, poor quality.", "output": "Negative"}
            ]

            example_prompt = PromptTemplate(
                input_variables=["input", "output"],
                template="Input: {input}\nOutput: {output}"
            )

            few_shot_prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix="Classify the sentiment of the following texts as Positive, Negative, or Neutral:",
                suffix="Input: {input}\nOutput:",
                input_variables=["input"]
            )

        elif task_type == "translation":
            # Few-shot examples for English to French translation
            examples = [
                {"input": "Hello", "output": "Bonjour"},
                {"input": "Thank you", "output": "Merci"},
                {"input": "How are you?", "output": "Comment allez-vous?"},
                {"input": "Good morning", "output": "Bonjour"},
                {"input": "See you later", "output": "À bientôt"}
            ]

            example_prompt = PromptTemplate(
                input_variables=["input", "output"],
                template="English: {input}\nFrench: {output}"
            )

            few_shot_prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix="Translate the following English phrases to French:",
                suffix="English: {input}\nFrench:",
                input_variables=["input"]
            )

        chain = few_shot_prompt | self.llm | StrOutputParser()
        return chain.invoke({"input": user_input})

    def structured_output(self, user_input: str) -> PersonalityResponse:
        """Generate structured output using Pydantic models."""

        parser = PydanticOutputParser(pydantic_object=PersonalityResponse)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that provides responses with personality analysis.

            Answer the user's question or prompt, then analyze your response.

            {format_instructions}

            CRITICAL: You MUST respond with ONLY valid JSON. Do not include:
            - <think> tags or reasoning
            - Any text before the JSON
            - Any text after the JSON
            - Code blocks or markdown

            Set confidence between 0.0 and 1.0 based on how certain you are about your answer."""),
            ("human", "{input}")
        ])

        # Create chain without parser, we'll handle parsing manually
        chain = (
            {"input": RunnablePassthrough(), "format_instructions": lambda _: parser.get_format_instructions()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            # Get raw output
            raw_output = chain.invoke(user_input)

            # Clean up the output - remove <think> tags and extract JSON
            import re
            # Remove <think>...</think> tags
            cleaned = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
            # Remove markdown code blocks
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
            # Strip whitespace
            cleaned = cleaned.strip()

            # Parse the cleaned JSON
            return parser.parse(cleaned)

        except Exception as e:
            # Fallback response if parsing fails
            return PersonalityResponse(
                response=f"I encountered an error processing your request. Try a more detailed query.",
                tone="apologetic",
                confidence=0.0
            )


def print_help():
    """Print the help message with all available commands."""
    print("\n📚 Available Commands:")
    print("=" * 50)
    print("  /role [role]        - Set personality role")
    print("                        Roles: helpful_assistant, creative_writer,")
    print("                               technical_expert, philosopher, comedian")
    print("  /cot on|off         - Enable/disable Chain of Thought reasoning")
    print("  /context set        - Set custom context (name, location, interests)")
    print("  /context off        - Disable context-aware mode")
    print("  /flow configure     - Configure conversation flow settings")
    print("  /flow status        - Show flow configuration")
    print("  /few_shot [task]    - Run few-shot learning task")
    print("                        Tasks: classification, translation")
    print("  /structured [query] - Get structured JSON output")
    print("  /status             - Show active features and settings")
    print("  /history            - Display all chat messages")
    print("  /clear              - Clear chat history")
    print("  /help               - Show this help message")
    print("  /quit               - Exit the chatbot")
    print("=" * 50)


def demo_advanced_features():
    """Run the advanced chatbot in interactive mode."""
    print("🧠 Advanced LangChain Chatbot")
    print("=" * 50)
    print("Features can be combined! Type /help for commands.")
    print("-" * 50)

    try:
        chatbot = AdvancedChatBot()
        print(f"✅ Initialized with role: helpful_assistant")

        while True:
            # Clear any buffered input before prompting
            import sys
            if sys.stdin.isatty():
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)

            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == '/quit':
                print("👋 Goodbye!")
                break

            if user_input.lower() == '/clear':
                chatbot.clear_history()
                continue

            if user_input.lower() == '/help':
                print_help()
                continue

            if user_input.lower() == '/history':
                history = chatbot.chat_history
                summary = chatbot.conversation_summary
                flow_config = chatbot.flow_config

                if not history and not summary:
                    print("📝 No chat history yet.")
                else:
                    print("\n📝 Chat History:")
                    print("=" * 50)

                    # Show summary first if it exists and summarization is enabled
                    if summary and flow_config["summarize_old"]:
                        print("\n[Previous conversation summary]")
                        print(f"{summary}")
                        print()

                    # Then show recent history
                    if history:
                        window_size = flow_config["window_size"]
                        if window_size > 0 and len(history) > window_size:
                            print(f"[Showing last {window_size} of {len(history)} exchanges]")
                            print("-" * 50)
                            display_history = history[-window_size:]
                            start_idx = len(history) - window_size + 1
                        else:
                            display_history = history
                            start_idx = 1

                        for i, exchange in enumerate(display_history, start_idx):
                            print(f"{i}. You: {exchange['user']}")
                            print(f"   Bot: {exchange['bot'][:100]}..." if len(exchange['bot']) > 100 else f"   Bot: {exchange['bot']}")
                            print()

                    print("=" * 50)
                continue

            if user_input.lower() == '/status':
                features = chatbot.features
                flow = chatbot.flow_config

                print("\n📊 Active Features:")
                print("=" * 50)
                print(f"🎭 Role: {features['role']}")
                print(f"🧠 Chain of Thought: {'✅ ON' if features['cot'] else '❌ OFF'}")
                if features['context']:
                    print(f"📍 Context: {features['context']}")
                else:
                    print("📍 Context: ❌ OFF")
                print(f"\n💬 Conversation Flow:")
                print(f"  - History: {len(chatbot.chat_history)}/{flow['max_history']} exchanges")
                print(f"  - Window Size: {flow['window_size']} exchanges")
                print(f"  - Topic Detection: {'✅ ON' if flow['detect_topics'] else '❌ OFF'}")
                print(f"  - Summarization: {'✅ ON' if flow['summarize_old'] else '❌ OFF'}")
                if chatbot.conversation_summary:
                    print(f"  - Summary exists: ✅")
                print("=" * 50)
                continue

            if user_input.lower().startswith('/role'):
                parts = user_input.split()
                role = parts[1] if len(parts) > 1 else "helpful_assistant"
                chatbot.set_role(role)
                continue

            if user_input.lower().startswith('/cot'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("⚠️  Usage: /cot on|off")
                    continue
                if parts[1].lower() == 'on':
                    chatbot.enable_cot()
                elif parts[1].lower() == 'off':
                    chatbot.disable_cot()
                else:
                    print("⚠️  Usage: /cot on|off")
                continue

            if user_input.lower().startswith('/context'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("⚠️  Usage: /context set | /context off")
                    continue
                if parts[1].lower() == 'set':
                    print("\n📝 Setting Context Information:")
                    print("-" * 50)
                    name = input("User Name: ").strip() or "User"
                    location = input("Location: ").strip() or "Unknown"
                    interests = input("Interests: ").strip() or "Various topics"

                    context = {
                        "user_name": name,
                        "location": location,
                        "interests": interests
                    }
                    chatbot.set_context(context)
                elif parts[1].lower() == 'off':
                    chatbot.clear_context()
                else:
                    print("⚠️  Usage: /context set | /context off")
                continue

            if user_input.lower().startswith('/flow'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("⚠️  Usage: /flow configure | /flow status")
                    continue

                if parts[1].lower() == 'configure':
                    print("\n⚙️  Configure Conversation Flow:")
                    print("-" * 50)

                    max_hist = input(f"Max history (current: {chatbot.flow_config['max_history']}): ").strip()
                    window = input(f"Window size (current: {chatbot.flow_config['window_size']}): ").strip()
                    topics = input(f"Detect topics? (current: {chatbot.flow_config['detect_topics']}) [yes/no]: ").strip().lower()
                    summarize = input(f"Summarize old messages? (current: {chatbot.flow_config['summarize_old']}) [yes/no]: ").strip().lower()

                    chatbot.configure_flow(
                        max_history=int(max_hist) if max_hist else None,
                        window_size=int(window) if window else None,
                        detect_topics=topics == 'yes' if topics else None,
                        summarize_old=summarize == 'yes' if summarize else None
                    )
                elif parts[1].lower() == 'status':
                    flow = chatbot.flow_config
                    print("\n💬 Conversation Flow Configuration:")
                    print("=" * 50)
                    print(f"  Max History: {flow['max_history']} exchanges")
                    print(f"  Window Size: {flow['window_size']} exchanges")
                    print(f"  Topic Detection: {'✅ ON' if flow['detect_topics'] else '❌ OFF'}")
                    print(f"  Summarization: {'✅ ON' if flow['summarize_old'] else '❌ OFF'}")
                    print(f"  Current History: {len(chatbot.chat_history)} exchanges")
                    if chatbot.conversation_summary:
                        print(f"\n📝 Summary: {chatbot.conversation_summary[:100]}...")
                    print("=" * 50)
                else:
                    print("⚠️  Usage: /flow configure | /flow status")
                continue

            if user_input.lower().startswith('/few_shot'):
                parts = user_input.split(maxsplit=2)
                if len(parts) < 3:
                    print("⚠️  Usage: /few_shot [classification|translation] [text]")
                    continue
                task_type = parts[1]
                text = parts[2]
                result = chatbot.few_shot_learning(text, task_type)
                print(f"🤖 Result: {result}")
                continue

            if user_input.lower().startswith('/structured'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("⚠️  Usage: /structured [query]")
                    print("💡 Example: /structured Tell me about artificial intelligence")
                    continue
                query = parts[1]
                try:
                    result = chatbot.structured_output(query)
                    print(f"\n📊 Structured Output:")
                    print("=" * 50)
                    print(f"📝 Response: {result.response}")
                    print(f"🎭 Tone: {result.tone}")
                    print(f"📈 Confidence: {result.confidence}")
                    print("=" * 50)
                except Exception as e:
                    print(f"❌ Error getting structured output: {e}")
                    print("💡 Tip: Try a more detailed question like 'Explain quantum computing'")
                continue

            # Check for unrecognized commands
            if user_input.startswith('/'):
                print(f"⚠️  Unrecognized command: {user_input}")
                print_help()
                continue

            # Process chat
            print("🤖 Bot: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_advanced_features()