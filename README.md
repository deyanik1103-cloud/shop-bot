# Gadget Hub Support Bot (RAG)

Customer support chatbot for a fictional online shop. Answers come from the shop's own documents (RAG) and live stock lookups (tool use), not from the model's imagination.

## Run
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key   # optional: without it the app runs in demo mode and shows retrieved text only
    python app.py                        # http://127.0.0.1:5000

## How it works
1. Retrieve: `knowledge/*.md` is split into chunks; TF-IDF finds the chunks closest to the question.
2. Augment: those chunks go into the system prompt as the only allowed source of truth.
3. Generate: Claude answers, and can call `get_stock` for live price and availability from `data/products.json`.

## Customise
Replace the files in `knowledge/` and `data/products.json` with a real shop's content.

## Roadmap
Embeddings + vector DB (Chroma), streaming replies, MySQL-backed products, retrieval evaluation set, guardrail tests, escalation to a human agent, Docker.
