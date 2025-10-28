import os
import json
import cohere
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
COHERE_API_KEY = os.getenv("CO_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "telecom_kb"

co = cohere.Client(COHERE_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

# -------------------------------
# 📘 1. Setup & Preload Data
# -------------------------------
def setup_collection(vector_size: int = 1024):
    try:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print(f" Collection '{COLLECTION_NAME}' initialized.")
    except Exception as e:
        print(f" Qdrant setup skipped: {e}")


def preload_telecom_data(json_path="data/telecom_kb.json"):
    print("preload_telcom_data invoked")
    with open(json_path, "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    embeddings = co.embed(
        texts=texts, model="embed-english-v3.0", input_type="search_document"
    ).embeddings

    setup_collection(vector_size=len(embeddings[0]))

    points = [
        models.PointStruct(id=item["id"], vector=vector, payload=item)
        for item, vector in zip(data, embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f" Uploaded {len(points)} telecom knowledge entries to Qdrant.")


# -------------------------------
# 🤖 2. Intent + Sentiment Detection
# -------------------------------
def detect_intent_and_sentiment(text: str):
    prompt = f"""
    Analyze this telecom message and return:
    - Intent: (plan_info, billing_issue, network_issue, sim_issue, service_activation, complaint, general_query)
    - Sentiment: (positive, neutral, negative)
    Message: "{text}"

    Respond strictly in JSON:
    {{"intent": "plan_info", "sentiment": "neutral"}}
    """

    try:
        response = co.chat(message=prompt, model="command-a-03-2025", temperature=0)
        result = json.loads(response.text.strip())
        intent = result.get("intent", "general_query")
        sentiment = result.get("sentiment", "neutral")
        print(f"[DEBUG] Classification → Intent: {intent}, Sentiment: {sentiment}")
        return intent, sentiment
    except Exception as e:
        print(f" Cohere classification error: {e}")
        return "general_query", "neutral"


# -------------------------------
#  3. Memory Summarization
# -------------------------------
def summarize_memory(history):
    """Condense past conversation into short context."""
    if not history:
        return ""

    chat_log = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[-6:]])
    prompt = f"""
    Summarize this telecom support conversation briefly, capturing key problems, actions, and tone:
    {chat_log}
    """

    try:
        response = co.chat(model="command-a-03-2025", message=prompt, temperature=0.2)
        return response.text.strip()
    except Exception as e:
        print(f" Memory summarization failed: {e}")
        return ""


# -------------------------------
# 🔍 4. Qdrant Similarity Search
# -------------------------------
def search_similar(query: str, top_k=3):
    print("\n [DEBUG] Searching similar entries in Qdrant...")
    embedding = co.embed(
        texts=[query], model="embed-english-v3.0", input_type="search_query"
    ).embeddings[0]

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME, query_vector=embedding, limit=top_k
    )
    return search_result


# -------------------------------
# 💬 5. Chat Response Generation
# -------------------------------
def generate_response(query: str, history=None, prev_sentiment="neutral", memory_context=""):
    history = history or []

    # Step 1: Detect sentiment and intent
    intent, sentiment = detect_intent_and_sentiment(query)

    if prev_sentiment == "negative" and sentiment == "neutral":
        sentiment = "negative"

    # Step 2: Retrieve context from Qdrant
    results = search_similar(query)
    context_text = "\n".join([r.payload.get("text", "") for r in results]) if results else "No relevant entries."

    # Step 3: Prepare conversation & memory
    conversation_context = "\n".join(
        [f"User: {m['user']}\nBot: {m['bot']}" for m in history[-3:]]
    )

    full_context = f"{memory_context}\n{conversation_context}".strip()

    # Step 4: Generate response
    prompt = f"""
You are a helpful telecom support chatbot.
Use previous memory to stay consistent and empathetic.
Context memory: {memory_context}

Conversation so far:
{conversation_context}

Current message: {query}
Detected intent: {intent}
User sentiment: {sentiment}

Relevant knowledge:
{context_text}

Now respond naturally and helpfully:
"""

    try:
        response = co.chat(
            model="command-a-03-2025", message=prompt, temperature=0.5, max_tokens=250
        )
        bot_reply = response.text.strip()

        history.append({"user": query, "bot": bot_reply})

        # Update memory every 3 messages
        if len(history) % 3 == 0:
            memory_context = summarize_memory(history)

        escalate = check_escalation_needed(query, sentiment, intent)
        handoff_summary = generate_handoff_summary(history) if escalate else None

        return {
            "reply": bot_reply,
            "sentiment": sentiment,
            "intent": intent,
            "history": history,
            "context": context_text,
            "memory_context": memory_context,
            "escalate": escalate,
            "handoff_summary": handoff_summary,
        }

    except Exception as e:
        print(f" Chat error: {e}")
        return {
            "reply": "Sorry, I'm having trouble right now. Please try again.",
            "history": history,
            "intent": "general_query",
            "sentiment": sentiment,
            "memory_context": memory_context,
        }


# -------------------------------
# 🧾 6. Escalation & Handoff
# -------------------------------
def check_escalation_needed(query, sentiment, intent):
    escalation_keywords = [
        "angry", "frustrated", "useless", "no help", "speak to agent", "human", "escalate", "complaint"
    ]
    if (sentiment == "negative" or intent == "complaint") and any(
        word in query.lower() for word in escalation_keywords
    ):
        return True
    return False


def generate_handoff_summary(history):
    if not history:
        return "No chat history available."

    chat_log = "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in history[-3:]])
    prompt = f"""
Summarize this chat for a human support agent.
Include: main issue, tone, and what the bot already advised.
Keep it concise.
{chat_log}
"""
    try:
        return co.chat(model="command-a-03-2025", message=prompt).text.strip()
    except Exception as e:
        print(f" Handoff summary error: {e}")
        return "Handoff summary could not be generated."
