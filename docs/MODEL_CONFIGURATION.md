# Model Configuration Guide

This guide explains how to configure and use different LLM providers with the LangChain AI Application.

## Overview

The application uses a unified model factory (`src/models/model_factory.py`) that supports three providers:

1. **OpenAI** - Cloud-based GPT models
2. **Ollama** - Local open-source models
3. **Qwen** - Alibaba Cloud's Qwen models via Dashscope API

## Quick Start

### 1. Copy the Example Configuration

```bash
cp .env.example .env
```

### 2. Choose Your Provider

Edit `.env` and set `MODEL_PROVIDER` to one of:
- `openai`
- `ollama`
- `qwen`

### 3. Test Your Configuration

```bash
python test_model_config.py
```

Or use the interactive setup:

```bash
python test_model_config.py --setup
```

## Provider-Specific Setup

### OpenAI

**Best for:** Production applications, highest quality responses

**Setup:**

1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)

2. Configure in `.env`:
```bash
MODEL_PROVIDER=openai
MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=sk-proj-...
MODEL_TEMPERATURE=0.7
```

**Available Models:**
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4` - Most capable
- `gpt-4-turbo` - Faster GPT-4
- `gpt-4o` - Latest optimized model

**Pricing:** Pay per token (see [OpenAI Pricing](https://openai.com/pricing))

---

### Ollama (Local)

**Best for:** Privacy, offline usage, no API costs

**Setup:**

1. Install Ollama:
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

2. Pull a model:
```bash
# Qwen3 (recommended)
ollama pull qwen3:latest

# Or other models
ollama pull llama2
ollama pull mistral
ollama pull codellama
```

3. Configure in `.env`:
```bash
MODEL_PROVIDER=ollama
MODEL_NAME=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434
MODEL_TEMPERATURE=0.7
```

**Popular Models:**
- `qwen3:latest` - Latest Qwen model (recommended)
- `llama2` - Meta's Llama 2
- `mistral` - Mistral 7B
- `codellama` - Code-focused
- `gemma` - Google's Gemma

**View all available models:**
```bash
ollama list          # Installed models
ollama search qwen   # Search for models
```

**Requirements:**
- 8GB+ RAM for 7B models
- 16GB+ RAM for 13B models
- 32GB+ RAM for larger models

---

### Qwen (Cloud API)

**Best for:** Access to latest Qwen models, Chinese language, long context

**Setup:**

1. Get an API key from [Dashscope Console](https://dashscope.console.aliyun.com/)
   - Register for Alibaba Cloud account
   - Enable Dashscope service
   - Create API key in console

2. Configure in `.env`:
```bash
MODEL_PROVIDER=qwen
MODEL_NAME=qwen-max
QWEN_API_KEY=sk-...
MODEL_TEMPERATURE=0.7
```

**Available Models:**

| Model | Context | Description | Best For |
|-------|---------|-------------|----------|
| `qwen-max` | 8K | Latest and most capable | General use, complex tasks |
| `qwen3-max` | 8K | Qwen 3 series flagship | Latest features |
| `qwen-plus` | 32K | Balanced performance | Cost-effective |
| `qwen-turbo` | 8K | Fast responses | High throughput |
| `qwen-long` | 10M | Ultra-long context | Long documents |
| `qwen2.5-72b-instruct` | 32K | Qwen 2.5 large | Advanced reasoning |
| `qwen2.5-32b-instruct` | 32K | Qwen 2.5 medium | Balanced |
| `qwen2.5-14b-instruct` | 8K | Qwen 2.5 small | Cost-effective |

**Features:**
- ✅ Excellent Chinese language support
- ✅ Extended context windows
- ✅ OpenAI-compatible API
- ✅ Competitive pricing

**Pricing:** See [Dashscope Pricing](https://dashscope.console.aliyun.com/)

---

## Configuration Examples

### Example 1: Development with Local Ollama

```bash
MODEL_PROVIDER=ollama
MODEL_NAME=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434
MODEL_TEMPERATURE=0.7
```

**Pros:** Free, private, offline
**Cons:** Requires local resources, slower than cloud

### Example 2: Production with OpenAI

```bash
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4-turbo
OPENAI_API_KEY=sk-proj-...
MODEL_TEMPERATURE=0.5
```

**Pros:** High quality, reliable, scalable
**Cons:** Costs per use, requires internet

### Example 3: Qwen3-max for Advanced Tasks

```bash
MODEL_PROVIDER=qwen
MODEL_NAME=qwen3-max
QWEN_API_KEY=sk-...
MODEL_TEMPERATURE=0.7
```

**Pros:** Latest features, good performance, Chinese support
**Cons:** Requires API key, China-based service

### Example 4: Long Context with Qwen-long

```bash
MODEL_PROVIDER=qwen
MODEL_NAME=qwen-long
QWEN_API_KEY=sk-...
MODEL_TEMPERATURE=0.5
```

**Use case:** Processing very long documents (up to 10M tokens)

---

## Temperature Settings

The `MODEL_TEMPERATURE` parameter controls response randomness:

- **0.0-0.3**: Deterministic, focused, factual
  - Best for: Code generation, factual Q&A, analysis

- **0.4-0.7**: Balanced creativity and focus (default: 0.7)
  - Best for: General chat, writing assistance

- **0.8-1.0**: Creative, diverse, exploratory
  - Best for: Creative writing, brainstorming

---

## Switching Between Providers

You can easily switch providers by changing the `.env` file:

```bash
# Edit .env
vim .env

# Or use test script
python test_model_config.py --setup

# Restart your application
python src/main.py
```

**No code changes required!** The model factory handles all provider differences.

---

## Programmatic Usage

### Using Default Configuration (from .env)

```python
from models import get_llm

# Automatically loads from environment variables
llm = get_llm()
response = llm.invoke("Hello!")
```

### Using Custom Configuration

```python
from models import get_llm, ModelConfig

# Create custom configuration
config = ModelConfig(
    provider="qwen",
    model_name="qwen-max",
    temperature=0.5,
    api_key="sk-..."
)

llm = get_llm(config)
response = llm.invoke("Hello!")
```

### Checking Current Configuration

```python
from models import print_model_info

print_model_info()
```

---

## Troubleshooting

### OpenAI Issues

**Error: "Invalid API key"**
- Verify key is correct in `.env`
- Check key hasn't expired at platform.openai.com
- Ensure no extra spaces in the key

**Error: "Rate limit exceeded"**
- You've hit your usage limit
- Upgrade your OpenAI plan
- Wait and retry later

### Ollama Issues

**Error: "Connection refused"**
- Ensure Ollama is running: `ollama serve`
- Check the URL in OLLAMA_BASE_URL
- Verify port 11434 is not blocked

**Error: "Model not found"**
- Pull the model: `ollama pull qwen3:latest`
- Check installed models: `ollama list`
- Use exact model name from `ollama list`

**Slow responses**
- Try a smaller model (7B instead of 13B)
- Close other applications to free RAM
- Consider using quantized models (e.g., `qwen3:7b-q4`)

### Qwen Issues

**Error: "Invalid API key"**
- Get key from dashscope.console.aliyun.com
- Set QWEN_API_KEY in `.env`
- Check key has correct permissions

**Error: "Model not found"**
- Verify model name spelling
- Check model availability in your region
- See list of models in this guide

---

## Best Practices

### 1. Development vs Production

**Development:**
- Use Ollama for free local testing
- Lower temperature for reproducibility
- Smaller models for faster iteration

**Production:**
- Use OpenAI or Qwen for reliability
- Monitor API costs
- Implement rate limiting
- Add error handling

### 2. Cost Optimization

- Use cheaper models (gpt-3.5-turbo, qwen-turbo) when possible
- Implement prompt caching
- Limit context window size
- Consider Ollama for high-volume use cases

### 3. Security

- Never commit `.env` to git
- Use environment variables in production
- Rotate API keys regularly
- Restrict API key permissions

### 4. Performance

- Use streaming for long responses
- Implement request timeouts
- Cache frequent queries
- Monitor response times

---

## Getting Help

- **OpenAI:** https://platform.openai.com/docs
- **Ollama:** https://ollama.ai/docs
- **Qwen/Dashscope:** https://dashscope.aliyun.com/docs

For issues with this application, check:
1. Run `python test_model_config.py` to diagnose
2. Review `.env.example` for correct format
3. Check the logs for detailed error messages
