"""
Common model factory for LangChain AI Application.
Supports OpenAI, Ollama, and Qwen models with configuration via environment variables.

Environment Variables:
    MODEL_PROVIDER: Model provider to use ('openai', 'ollama', or 'qwen')
    MODEL_NAME: Specific model name (provider-specific)
    MODEL_TEMPERATURE: Temperature for response generation (0.0-1.0)
    EMBEDDING_PROVIDER: Embedding provider to use ('openai', 'ollama', or 'dashscope')
    EMBEDDING_MODEL: Specific embedding model name (provider-specific)
    OPENAI_API_KEY: API key for OpenAI models
    OLLAMA_BASE_URL: Base URL for Ollama server (default: http://localhost:11434)
    QWEN_API_KEY: API key for Qwen models (Dashscope API key)
    QWEN_BASE_URL: Base URL for Qwen API (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
    DASHSCOPE_API_KEY: API key for Dashscope embeddings
"""

import os
from typing import Optional, Literal
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    provider: Literal["openai", "ollama", "qwen"]
    model_name: Optional[str] = None
    temperature: float = 0.7
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # Embedding model configuration
    embedding_provider: Optional[Literal["openai", "ollama", "dashscope"]] = None
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create ModelConfig from environment variables."""
        provider = os.getenv("MODEL_PROVIDER", "ollama").lower()

        if provider not in ["openai", "ollama", "qwen"]:
            raise ValueError(
                f"Invalid MODEL_PROVIDER: {provider}. Must be 'openai', 'ollama', or 'qwen'"
            )

        # Get common settings
        temperature = float(os.getenv("MODEL_TEMPERATURE", "0.7"))

        # Provider-specific defaults
        if provider == "openai":
            model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")  # Optional custom endpoint
        elif provider == "ollama":
            model_name = os.getenv("MODEL_NAME", "qwen3:latest")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            api_key = None
        elif provider == "qwen":
            model_name = os.getenv("MODEL_NAME", "qwen-max")
            api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

        # Embedding model configuration (optional)
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
        if embedding_provider not in ["openai", "ollama", "dashscope"]:
            embedding_provider = "ollama"  # Default fallback

        embedding_model = os.getenv("EMBEDDING_MODEL")
        if not embedding_model:
            # Set defaults based on provider
            if embedding_provider == "openai":
                embedding_model = "text-embedding-3-small"
            elif embedding_provider == "ollama":
                embedding_model = "nomic-embed-text"
            elif embedding_provider == "dashscope":
                embedding_model = "text-embedding-v3"

        # Get embedding API key
        embedding_api_key = None
        if embedding_provider == "openai":
            embedding_api_key = os.getenv("OPENAI_API_KEY")
        elif embedding_provider == "dashscope":
            embedding_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")

        return cls(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key
        )


def get_llm(config: Optional[ModelConfig] = None):
    """
    Get a Language Model instance based on configuration.

    Args:
        config: ModelConfig instance. If None, loads from environment variables.

    Returns:
        LLM instance (OllamaLLM or ChatOpenAI compatible)

    Raises:
        ValueError: If required configuration is missing or invalid
    """
    if config is None:
        config = ModelConfig.from_env()

    provider = config.provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not config.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it to use OpenAI models."
            )

        print(f"🤖 Using OpenAI model: {config.model_name}")

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": config.api_key,
        }

        if config.base_url:
            kwargs["base_url"] = config.base_url

        return ChatOpenAI(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import OllamaLLM

        base_url = config.base_url or "http://localhost:11434"

        print(f"🔧 Connecting to Ollama server: {base_url}")
        print(f"🤖 Using Ollama model: {config.model_name}")

        return OllamaLLM(
            model=config.model_name,
            base_url=base_url,
            temperature=config.temperature
        )

    elif provider == "qwen":
        from langchain_openai import ChatOpenAI

        if not config.api_key:
            raise ValueError(
                "QWEN_API_KEY (or DASHSCOPE_API_KEY) not found in environment variables. "
                "Please set it to use Qwen models. Get your API key from: "
                "https://dashscope.console.aliyun.com/"
            )

        base_url = config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        print(f"🚀 Using Qwen model: {config.model_name}")
        print(f"🔧 API endpoint: {base_url}")

        # Qwen uses OpenAI-compatible API
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=base_url
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_chat_llm(config: Optional[ModelConfig] = None):
    """
    Get a Chat Language Model instance (alias for get_llm for clarity).

    Args:
        config: ModelConfig instance. If None, loads from environment variables.

    Returns:
        Chat LLM instance
    """
    return get_llm(config)


def get_embeddings(config: Optional[ModelConfig] = None):
    """
    Get an Embeddings instance based on configuration.

    Args:
        config: ModelConfig instance. If None, loads from environment variables.

    Returns:
        Embeddings instance

    Raises:
        ValueError: If required configuration is missing or invalid
    """
    if config is None:
        config = ModelConfig.from_env()

    provider = config.embedding_provider or "ollama"
    model = config.embedding_model

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not config.embedding_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it to use OpenAI embeddings."
            )

        print(f"📊 Using OpenAI embeddings: {model}")
        return OpenAIEmbeddings(
            model=model,
            api_key=config.embedding_api_key
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        base_url = config.base_url or "http://localhost:11434"
        print(f"📊 Using Ollama embeddings: {model}")

        return OllamaEmbeddings(
            model=model,
            base_url=base_url
        )

    elif provider == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings

        if not config.embedding_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY (or QWEN_API_KEY) not found in environment variables. "
                "Please set it to use Dashscope embeddings. Get your API key from: "
                "https://dashscope.console.aliyun.com/"
            )

        print(f"📊 Using Dashscope embeddings: {model}")

        return DashScopeEmbeddings(
            model=model,
            dashscope_api_key=config.embedding_api_key
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def print_model_info():
    """Print current model configuration from environment."""
    try:
        config = ModelConfig.from_env()
        print("\n" + "=" * 60)
        print("📋 Current Model Configuration")
        print("=" * 60)
        print(f"Provider:    {config.provider}")
        print(f"Model:       {config.model_name}")
        print(f"Temperature: {config.temperature}")
        if config.base_url:
            print(f"Base URL:    {config.base_url}")
        if config.provider == "openai":
            print(f"API Key:     {'✓ Set' if config.api_key else '✗ Not Set'}")
        elif config.provider == "qwen":
            print(f"API Key:     {'✓ Set' if config.api_key else '✗ Not Set'}")
        print("\n" + "-" * 60)
        print("📊 Embedding Configuration")
        print("-" * 60)
        print(f"Provider:    {config.embedding_provider}")
        print(f"Model:       {config.embedding_model}")
        if config.embedding_provider in ["openai", "dashscope"]:
            print(f"API Key:     {'✓ Set' if config.embedding_api_key else '✗ Not Set'}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"❌ Error loading model configuration: {e}")


if __name__ == "__main__":
    # Test the model factory
    print_model_info()

    try:
        llm = get_llm()
        print("✅ LLM loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading LLM: {e}")

    try:
        embeddings = get_embeddings()
        print("✅ Embeddings loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
