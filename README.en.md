# aiwolf-nlp-agent-llm

[README in Japanese](/README.md)

This is a sample agent using LLM for the AIWolf Competition (Natural Language Division).

## Environment Setup

> [!IMPORTANT]
> Python 3.11 or higher is required.

### Using uv

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
uv sync
```

When using uv, replace `python src/main.py` with `uv run src/main.py` in the instructions below.

### Without uv

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### If you would like to use prompts written in Japanese
```bash
cp config/config.jp.yml.example config/config.yml
```

### If you would like to use prompts written in English
```bash
cp config/config.en.yml.example config/config.yml
```

## Others

For details on execution methods, settings, and other information, please refer to [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent).
