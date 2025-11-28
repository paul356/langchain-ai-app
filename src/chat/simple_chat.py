"""
Basic chat application using LangChain.
Supports both OpenAI and local models via Ollama.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class SimpleChatBot:
    """A simple chatbot using LangChain with configurable LLM backends."""
    
    def __init__(self, model_type: str = "openai", model_name: Optional[str] = None):
        """
        Initialize the chatbot.
        
        Args:
            model_type: Either "openai" or "ollama"
            model_name: Specific model name (optional)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.llm = self._setup_llm()
        self.chat_history = []
        
    def _setup_llm(self):
        """Set up the language model based on configuration."""
        if self.model_type == "openai":
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            model = self.model_name or "gpt-3.5-turbo"
            return ChatOpenAI(
                model=model,
                temperature=0.7,
                api_key=api_key
            )
        
        elif self.model_type == "ollama":
            from langchain_ollama import OllamaLLM
            
            model = self.model_name or os.getenv("MODEL_NAME", "qwen3:latest")
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            print(f"🔧 Connecting to Ollama server: {ollama_url}")
            print(f"🤖 Using model: {model}")
            
            return OllamaLLM(
                model=model,
                base_url=ollama_url
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
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
    
    # Try to determine which model to use - prefer Ollama if configured
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    if ollama_url:
        print(f"🔧 Found Ollama configuration: {ollama_url}")
        model_type = "ollama"
    elif os.getenv("OPENAI_API_KEY"):
        print("🔧 Using OpenAI API")
        model_type = "openai"
    else:
        print("⚠️  No API key or Ollama configuration found. Trying default Ollama...")
        model_type = "ollama"
    
    try:
        # Initialize chatbot
        chatbot = SimpleChatBot(model_type=model_type)
        print(f"✅ Successfully initialized {model_type} model")
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
    main()