import json, os, re, time
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from rag import load_chunks, Retriever

app = Flask(__name__, static_folder="static", static_url_path="")
retriever = Retriever(load_chunks())
with open("data/products.json", encoding="utf-8") as f:
    PRODUCTS = json.load(f)

# Any OpenAI-compatible provider works. Defaults are Groq's free tier.
API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

SYSTEM = """You are the customer support assistant for Gadget Hub, an online electronics shop.
Answer ONLY using the CONTEXT below and the get_stock tool. If the answer is not there, say you
don't know and suggest emailing support@gadgethub.example. Be concise and friendly.
For any question about a product's price or stock, always call get_stock first.
Never invent policies, prices, or stock levels."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_stock",
        "description": "Look up the live price and stock level of a product by name.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
}]

STOP = {"the", "a", "an", "of", "for", "and", "price", "much", "how", "is", "in", "stock", "do", "you", "have"}

# Simple abuse protection so a public demo can't drain the free quota
PER_IP_LIMIT, WINDOW_SECONDS, DAILY_LIMIT = 10, 600, 300
HITS = defaultdict(deque)
DAILY = {"day": "", "count": 0}


def allowed(ip):
    now = time.time()
    q = HITS[ip]
    while q and now - q[0] > WINDOW_SECONDS:
        q.popleft()
    today = time.strftime("%Y-%m-%d")
    if DAILY["day"] != today:
        DAILY["day"], DAILY["count"] = today, 0
    if len(q) >= PER_IP_LIMIT or DAILY["count"] >= DAILY_LIMIT:
        return False
    q.append(now)
    DAILY["count"] += 1
    return True


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def get_stock(name):
    q = words(name)
    best, best_score = None, 0
    for p in PRODUCTS:
        score = len(q & words(p["name"]))
        if score > best_score:
            best, best_score = p, score
    if not best:
        return {"error": "Product not found"}
    return {"name": best["name"], "price_usd": best["price"], "in_stock": best["stock"] > 0, "units_left": best["stock"]}


def run_tool(tc):
    try:
        args = json.loads(tc.function.arguments or "{}")
    except ValueError:
        args = {}
    return json.dumps(get_stock(str(args.get("name", ""))))


@app.post("/api/chat")
def chat():
    msgs = [m for m in (request.json or {}).get("messages", [])[-10:]
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    if not msgs or msgs[-1]["role"] != "user":
        return jsonify(error="No question"), 400
    convo = [{"role": m["role"], "content": m["content"][:1000]} for m in msgs]

    hits = retriever.search(convo[-1]["content"])          # 1. RETRIEVE
    sources = sorted({h["source"] for h in hits})
    context = "\n\n---\n\n".join(h["text"] for h in hits) or "(no relevant documents found)"

    if not API_KEY:                                         # works offline in demo mode
        return jsonify(reply="(Demo mode: no API key set) " + context, sources=sources)

    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
    if not allowed(ip):
        return jsonify(error="Demo limit reached. Please try again later."), 429

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30, max_retries=2)
    messages = [{"role": "system", "content": SYSTEM + "\n\nCONTEXT:\n" + context}] + convo   # 2. AUGMENT
    try:
        for _ in range(3):                                  # 3. GENERATE (with a tool-use loop)
            r = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, max_tokens=500)
            msg = r.choices[0].message
            if not msg.tool_calls:
                break
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": run_tool(tc)})
    except Exception as e:
        app.logger.error("LLM error: %s", e)
        return jsonify(error="The assistant is unavailable right now."), 502
    return jsonify(reply=msg.content or "Sorry, I couldn't finish that. Please try again.", sources=sources)


@app.get("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(debug=True)