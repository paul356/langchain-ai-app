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
    FewShotPromptTemplate
)
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
    
    def role_based_chat(self, user_input: str, role: str = "helpful_assistant") -> str:
        """Chat with different AI personalities/roles."""
        
        role_prompts = {
            "helpful_assistant": """You are a helpful and knowledgeable assistant. 
            Provide clear, accurate, and friendly responses.""",
            
            "creative_writer": """You are a creative writer and storyteller. 
            Respond with imagination, vivid descriptions, and engaging narratives.""",
            
            "technical_expert": """You are a technical expert and engineer. 
            Provide precise, detailed, and technically accurate responses with examples.""",
            
            "philosopher": """You are a wise philosopher. 
            Respond with deep thoughts, ask meaningful questions, and explore ideas thoroughly.""",
            
            "comedian": """You are a friendly comedian. 
            Respond with humor, wit, and light-heartedness while still being helpful."""
        }
        
        system_prompt = role_prompts.get(role, role_prompts["helpful_assistant"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"input": user_input})
    
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
    
    def chain_of_thought(self, problem: str) -> str:
        """Implement chain-of-thought reasoning."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert problem solver. When given a problem, think through it step by step.
            
            Format your response as:
            
            **Problem Analysis:**
            [Break down the problem]
            
            **Step-by-step Solution:**
            1. [First step and reasoning]
            2. [Second step and reasoning]
            3. [Continue as needed]
            
            **Final Answer:**
            [Your conclusion]
            
            Think carefully and show your reasoning at each step."""),
            ("human", "{problem}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"problem": problem})
    
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
    
    def context_aware_chat(self, user_input: str, context: Dict[str, Any]) -> str:
        """Chat with additional context information."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent assistant. Use the provided context to give more relevant and personalized responses.
            
            Context Information:
            - User Name: {user_name}
            - Location: {location}
            - Interests: {interests}
            - Previous Topic: {previous_topic}
            
            Use this context naturally in your response when relevant."""),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Prepare context with defaults
        context_with_defaults = {
            "user_name": context.get("user_name", "User"),
            "location": context.get("location", "Unknown"),
            "interests": context.get("interests", "Various topics"),
            "previous_topic": context.get("previous_topic", "None"),
            "input": user_input
        }
        
        return chain.invoke(context_with_defaults)


def demo_advanced_features():
    """Demonstrate advanced prompting features."""
    print("🧠 Advanced LangChain Prompting Demo")
    print("=" * 50)
    
    try:
        chatbot = AdvancedChatBot()
        
        # 1. Role-based chat
        print("\n1. 🎭 Role-based Chat:")
        print("-" * 30)
        user_question = "How can I improve my programming skills?"
        
        roles = ["helpful_assistant", "technical_expert", "philosopher"]
        for role in roles:
            print(f"\n{role.replace('_', ' ').title()}:")
            response = chatbot.role_based_chat(user_question, role)
            print(response[:200] + "..." if len(response) > 200 else response)
        
        # 2. Few-shot learning
        print("\n\n2. 🎯 Few-shot Learning:")
        print("-" * 30)
        
        # Sentiment classification
        print("\nSentiment Classification:")
        text = "This movie was incredible! The acting was superb."
        result = chatbot.few_shot_learning(text, "classification")
        print(f"Text: {text}")
        print(f"Sentiment: {result}")
        
        # Translation
        print("\nTranslation:")
        text = "I am learning Python"
        result = chatbot.few_shot_learning(text, "translation")
        print(f"English: {text}")
        print(f"French: {result}")
        
        # 3. Chain of thought
        print("\n\n3. 🔗 Chain of Thought Reasoning:")
        print("-" * 30)
        problem = "If I have 15 apples and give away 1/3 of them, then buy 8 more apples, how many apples do I have?"
        response = chatbot.chain_of_thought(problem)
        print(response)
        
        # 4. Structured output
        print("\n\n4. 📊 Structured Output:")
        print("-" * 30)
        try:
            response = chatbot.structured_output("Tell me about artificial intelligence")
            print(f"Response: {response.response[:100]}...")
            print(f"Tone: {response.tone}")
            print(f"Confidence: {response.confidence}")
        except Exception as e:
            print(f"Structured output failed: {e}")
        
        # 5. Context-aware chat
        print("\n\n5. 🎯 Context-aware Chat:")
        print("-" * 30)
        context = {
            "user_name": "Alice",
            "location": "San Francisco",
            "interests": "Machine learning, hiking, photography",
            "previous_topic": "Python programming"
        }
        response = chatbot.context_aware_chat(
            "What should I do this weekend?", 
            context
        )
        print(response)
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        print("Make sure your API keys are set up correctly!")


if __name__ == "__main__":
    demo_advanced_features()