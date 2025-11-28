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
        """Format chat history into a list of messages."""
        messages = []
        for exchange in self.chat_history:
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
        print("🧹 History cleared.")

    def chat(self, user_input: str) -> str:
        """Main chat entry point that builds prompt from active features."""
        try:
            response = self._build_and_invoke(user_input)
            self.chat_history.append({"user": user_input, "bot": response})
            return response
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def _build_and_invoke(self, user_input: str) -> str:
        """Build prompt dynamically based on active features."""
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
    print("  /context on|off     - Enable/disable context-aware mode")
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
                if not history:
                    print("📝 No chat history yet.")
                else:
                    print("\n📝 Chat History:")
                    print("-" * 50)
                    for i, exchange in enumerate(history, 1):
                        print(f"{i}. You: {exchange['user']}")
                        print(f"   Bot: {exchange['bot'][:100]}..." if len(exchange['bot']) > 100 else f"   Bot: {exchange['bot']}")
                        print()
                continue

            if user_input.lower() == '/status':
                features = chatbot.features

                print("\n📊 Active Features:")
                print("=" * 50)
                print(f"🎭 Role: {features['role']}")
                print(f"🧠 Chain of Thought: {'✅ ON' if features['cot'] else '❌ OFF'}")
                if features['context']:
                    print(f"📍 Context: {features['context']}")
                else:
                    print("📍 Context: ❌ OFF")
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
                    print("⚠️  Usage: /context on|off")
                    continue
                if parts[1].lower() == 'on':
                    context = {
                        "user_name": "Alice",
                        "location": "Wonderland",
                        "interests": "Exploration, Riddles"
                    }
                    chatbot.set_context(context)
                elif parts[1].lower() == 'off':
                    chatbot.clear_context()
                else:
                    print("⚠️  Usage: /context on|off")
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