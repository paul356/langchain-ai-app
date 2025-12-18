"""
Basic chat application using LangChain.
Supports OpenAI, Ollama, and Qwen models via unified model factory.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import common model factory
from models import get_llm, ModelConfig


class SimpleChatBot:
    """A simple chatbot using LangChain with configurable LLM backends."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the chatbot.
        
        Args:
            config: ModelConfig instance. If None, loads from environment variables.
        """
        self.config = config or ModelConfig.from_env()
        self.llm = get_llm(self.config)
        self.chat_history = []
    
    def _format_history(self):
        """Format chat history into a list of messages."""
        messages = []
        for exchange in self.chat_history:
            messages.append(HumanMessage(content=exchange["user"]))
            messages.append(AIMessage(content=exchange["bot"]))
        return messages

    def create_chat_chain(self):
        """Create the chat processing chain."""
        
        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. You are knowledgeable, friendly, and concise. 
            Provide helpful and accurate responses to user questions."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        chain = (
            {
                "input": RunnablePassthrough(),
                "history": lambda x: self._format_history()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def chat(self, message: str) -> str:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: User input message
            
        Returns:
            Bot response
        """
        chain = self.create_chat_chain()
        
        try:
            # Get response from the chain
            response = chain.invoke(message)
            
            # Store in chat history
            self.chat_history.append({"user": message, "bot": response})
            
            return response
        
        except Exception as e:
            error_msg = f"Error getting response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_chat_history(self) -> list:
        """Get the chat history."""
        return self.chat_history
    
    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []


def main():
    """Main function to run the chatbot interactively."""
    print("🤖 Welcome to LangChain Chat!")
    print("Type 'quit' to exit, 'clear' to clear history, 'history' to see chat history")
    print()
    
    try:
        # Initialize chatbot with environment configuration
        chatbot = SimpleChatBot()
        print(f"✅ Model initialized successfully")
        print()
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🧹 Chat history cleared!")
                continue
            elif user_input.lower() == 'history':
                history = chatbot.get_chat_history()
                if not history:
                    print("📝 No chat history yet.")
                else:
                    print("\n📝 Chat History:")
                    for i, exchange in enumerate(history, 1):
                        print(f"{i}. You: {exchange['user']}")
                        print(f"   Bot: {exchange['bot']}")
                        print()
                continue
            elif not user_input:
                continue
            
            # Get response from chatbot
            print("🤖 Bot: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
            print()
    
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print()
        print("💡 Make sure to:")
        print("1. Set your OPENAI_API_KEY in .env file, OR")
        print("2. Install and run Ollama with a model (e.g., 'ollama pull llama2')")


if __name__ == "__main__":
    main()    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print()
        print("💡 Configuration options:")
        print("1. OpenAI: Set MODEL_PROVIDER=openai and OPENAI_API_KEY in .env")
        print("2. Ollama: Set MODEL_PROVIDER=ollama and install Ollama")
        print("3. Qwen: Set MODEL_PROVIDER=qwen and QWEN_API_KEY in .env")
        print()
        print("See .env.example for detailed configuration options.")