# aiwolf-nlp-agent-llm

[README in English](/README.en.md)

人狼知能コンテスト（自然言語部門） のLLMを用いたサンプルエージェントです。

## 環境構築

> [!IMPORTANT]
> Python 3.11以上が必要です。

### uvを使用する場合

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
uv sync
```

uvを使用する場合、以下の`python src/main.py`などを`uv run src/main.py`と読み替えてください。

### uvを使用しない場合

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 日本語のプロンプトを使用したい場合
```bash
cp config/config.jp.yml.example config/config.yml
```

### 英語のプロンプトを使用したい場合
```bash
cp config/config.en.yml.example config/config.yml
```

## その他

実行方法や設定などその他については[aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent)をご確認ください。
