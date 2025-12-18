# Quick Reference: Model Configuration

## One-Command Setup

```bash
# Interactive setup wizard
python test_model_config.py --setup
```

## Manual Setup

### 1. OpenAI
```bash
echo "MODEL_PROVIDER=openai
MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=sk-your-key-here
MODEL_TEMPERATURE=0.7" > .env
```

### 2. Ollama (Local)
```bash
ollama pull qwen3:latest
echo "MODEL_PROVIDER=ollama
MODEL_NAME=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434
MODEL_TEMPERATURE=0.7" > .env
```

### 3. Qwen Cloud
```bash
echo "MODEL_PROVIDER=qwen
MODEL_NAME=qwen-max
QWEN_API_KEY=sk-your-key-here
MODEL_TEMPERATURE=0.7" > .env
```

## Test Configuration

```bash
python test_model_config.py
```

## Quick Model Comparison

| Provider | Cost | Speed | Privacy | Quality |
|----------|------|-------|---------|---------|
| OpenAI | 💰💰💰 | ⚡⚡⚡ | ⚠️ Cloud | ⭐⭐⭐⭐⭐ |
| Ollama | 🆓 Free | ⚡⚡ | 🔒 Local | ⭐⭐⭐⭐ |
| Qwen | 💰💰 | ⚡⚡⚡ | ⚠️ Cloud | ⭐⭐⭐⭐⭐ |

## Recommended Models

**Best Overall:** `qwen-max` (Qwen Cloud)
**Best Free:** `qwen3:latest` (Ollama)
**Best Quality:** `gpt-4-turbo` (OpenAI)
**Best for Code:** `codellama` (Ollama)
**Best for Long Context:** `qwen-long` (Qwen Cloud)

## Get API Keys

- **OpenAI:** https://platform.openai.com/api-keys
- **Qwen:** https://dashscope.console.aliyun.com/

## Common Issues

### "Connection refused" → Start Ollama: `ollama serve`
### "Invalid API key" → Check key in `.env` file
### "Model not found" → Pull model: `ollama pull <model-name>`

## Full Documentation

See [docs/MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for complete guide.
