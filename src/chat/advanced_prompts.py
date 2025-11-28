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
        # Default to role mode with helpful_assistant
        self.current_mode = "role"
        self.mode_settings = {"role": "helpful_assistant"}
    
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

    def set_mode(self, mode: str, **kwargs):
        """Set the current chat mode and settings."""
        if mode == "standard":
            mode = "role"
            kwargs = {"role": "helpful_assistant"}
            
        self.current_mode = mode
        self.mode_settings = kwargs
        print(f"🔄 Switched to '{mode}' mode.")

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        print("🧹 History cleared.")

    def chat(self, user_input: str) -> str:
        """Main chat entry point that routes to specific handlers."""
        try:
            if self.current_mode == "role":
                response = self._chat_role(user_input)
            elif self.current_mode == "cot":
                response = self._chat_cot(user_input)
            elif self.current_mode == "context":
                response = self._chat_context(user_input)
            elif self.current_mode == "few_shot":
                # Few shot is usually stateless/task-based, but we can wrap it
                response = self.few_shot_learning(user_input, self.mode_settings.get("task_type", "classification"))
            elif self.current_mode == "structured":
                # Structured output returns an object, we'll format it to string
                result = self.structured_output(user_input)
                response = f"Response: {result.response}\nTone: {result.tone}\nConfidence: {result.confidence}"
            else:
                # Default fallback
                response = self._chat_role(user_input)

            # Store history (except for maybe structured/few-shot if we want to keep them separate, but let's store all for now)
            self.chat_history.append({"user": user_input, "bot": response})
            return response

        except Exception as e:
            return f"Error processing request: {str(e)}"

    def _chat_role(self, user_input: str) -> str:
        """Role-based chat with history."""
        role = self.mode_settings.get("role", "helpful_assistant")
        
        role_prompts = {
            "helpful_assistant": "You are a helpful and knowledgeable assistant. Provide clear, accurate, and friendly responses.",
            "creative_writer": "You are a creative writer and storyteller. Respond with imagination, vivid descriptions, and engaging narratives.",
            "technical_expert": "You are a technical expert and engineer. Provide precise, detailed, and technically accurate responses with examples.",
            "philosopher": "You are a wise philosopher. Respond with deep thoughts, ask meaningful questions, and explore ideas thoroughly.",
            "comedian": "You are a friendly comedian. Respond with humor, wit, and light-heartedness while still being helpful."
        }
        
        system_prompt = role_prompts.get(role, role_prompts["helpful_assistant"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        chain = (
            {
                "input": RunnablePassthrough(),
                "history": lambda x: self._format_history()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(user_input)

    def _chat_cot(self, user_input: str) -> str:
        """Chain of thought chat."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert problem solver. When given a problem, think through it step by step.
            
            Format your response as:
            **Problem Analysis:** [Break down the problem]
            **Step-by-step Solution:** [Numbered steps]
            **Final Answer:** [Conclusion]
            
            Think carefully and show your reasoning at each step."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        chain = (
            {
                "input": RunnablePassthrough(),
                "history": lambda x: self._format_history()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(user_input)

    def _chat_context(self, user_input: str) -> str:
        """Context-aware chat."""
        context = self.mode_settings.get("context", {})
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent assistant. Use the provided context to give more relevant and personalized responses.
            
            Context Information:
            - User Name: {user_name}
            - Location: {location}
            - Interests: {interests}
            
            Use this context naturally in your response when relevant."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Prepare context with defaults
        input_dict = {
            "user_name": context.get("user_name", "User"),
            "location": context.get("location", "Unknown"),
            "interests": context.get("interests", "Various topics"),
            "input": user_input
        }
        
        chain = (
            {
                "input": lambda x: x["input"],
                "user_name": lambda x: x["user_name"],
                "location": lambda x: x["location"],
                "interests": lambda x: x["interests"],
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
            
            {format_instructions}
            
            Analyze your response and provide the confidence level based on how certain you are about your answer."""),
            ("human", "{input}")
        ])
        
        chain = (
            {"input": RunnablePassthrough(), "format_instructions": lambda _: parser.get_format_instructions()}
            | prompt
            | self.llm
            | parser
        )
        
        return chain.invoke(user_input)


def print_help():
    """Print the help message with all available commands."""
    print("\n📚 Available Commands:")
    print("=" * 50)
    print("  /mode role [role]   - Switch to role mode")
    print("                        Roles: helpful_assistant, creative_writer,")
    print("                               technical_expert, philosopher, comedian")
    print("  /mode cot           - Chain of Thought reasoning mode")
    print("  /mode context       - Context-aware chat with user metadata")
    print("  /mode few_shot      - Few-shot learning mode")
    print("                        Tasks: classification, translation")
    print("  /mode structured    - Structured JSON output mode")
    print("  /mode standard      - Standard chat mode (alias for helpful_assistant)")
    print("  /status             - Show current mode and settings")
    print("  /history            - Display all chat messages")
    print("  /clear              - Clear chat history")
    print("  /help               - Show this help message")
    print("  /quit               - Exit the chatbot")
    print("=" * 50)


def demo_advanced_features():
    """Run the advanced chatbot in interactive mode."""
    print("🧠 Advanced LangChain Chatbot")
    print("=" * 50)
    print("Commands:")
    print("  /mode role [role]   - Switch to role mode (helpful_assistant, creative_writer, technical_expert, philosopher, comedian)")
    print("  /mode cot           - Switch to Chain of Thought mode")
    print("  /mode context       - Switch to Context-Aware mode")
    print("  /mode few_shot      - Switch to Few-Shot Learning mode")
    print("  /mode structured    - Switch to Structured Output mode")
    print("  /mode standard      - Switch to Standard Chat mode")
    print("  /status             - Show current mode and settings")
    print("  /history            - Show chat history")
    print("  /clear              - Clear chat history")
    print("  /help               - Show this help message")
    print("  /quit               - Exit")
    print("-" * 50)
    
    try:
        chatbot = AdvancedChatBot()
        print(f"✅ Initialized in 'role' mode (helpful_assistant).")
        
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
                mode = chatbot.current_mode
                settings = chatbot.mode_settings
                
                print(f"📊 Current Mode: {mode.upper()}")
                
                if mode == "role":
                    role = settings.get('role', 'unknown')
                    print(f"ℹ️  Description: Persona-based chat. Active Role: {role}")
                elif mode == "cot":
                    print("ℹ️  Description: Chain of Thought. The model explains its reasoning step-by-step.")
                elif mode == "context":
                    print("ℹ️  Description: Context-aware chat. Injects hidden user metadata into the prompt.")
                elif mode == "few_shot":
                    task = settings.get('task_type', 'unknown')
                    print(f"ℹ️  Description: Few-shot learning. Task: {task}")
                elif mode == "structured":
                    print("ℹ️  Description: Structured Output. Returns parsed JSON with 'response', 'tone', and 'confidence'.")
                
                if settings:
                    print(f"⚙️  Raw Settings: {settings}")
                continue
                
            if user_input.lower().startswith('/mode'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("⚠️  Usage: /mode [mode_name] [args]")
                    print_help()
                    continue
                
                mode = parts[1]
                
                if mode == 'role':
                    role = parts[2] if len(parts) > 2 else "helpful_assistant"
                    chatbot.set_mode("role", role=role)
                elif mode == 'cot':
                    chatbot.set_mode("cot")
                elif mode == 'context':
                    # For demo purposes, we'll use a fixed context
                    context = {
                        "user_name": "Alice",
                        "location": "Wonderland",
                        "interests": "Exploration, Riddles"
                    }
                    chatbot.set_mode("context", context=context)
                    print(f"Context set to: {context}")
                elif mode == 'few_shot':
                    task = parts[2] if len(parts) > 2 else "classification"
                    chatbot.set_mode("few_shot", task_type=task)
                    print(f"Task type set to: {task}")
                elif mode == 'structured':
                    chatbot.set_mode("structured")
                elif mode == 'standard':
                    chatbot.set_mode("standard")
                else:
                    print(f"⚠️  Unknown mode: {mode}")
                    print_help()
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