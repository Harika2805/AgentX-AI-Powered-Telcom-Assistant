import streamlit as st
from llm_pipeline import generate_response, preload_telecom_data

st.set_page_config(page_title="AgentX ", page_icon="📱")
st.image("logo.png", width=150)

# ✅ Load knowledge base once
if "data_loaded" not in st.session_state:
    preload_telecom_data()
    st.session_state.data_loaded = True
    st.success("👋 Hi there! I’m AgentX, your AI powered telecom assistant.!")

# ✅ Initialize memory and history
if "history" not in st.session_state:
    st.session_state.history = []
if "last_sentiment" not in st.session_state:
    st.session_state.last_sentiment = "neutral"
if "memory_context" not in st.session_state:
    st.session_state.memory_context = ""

# --- Display prior messages
for msg in st.session_state.history:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["bot"])

# --- Handle new message
user_input = st.chat_input("Ask your telecom question...")

if user_input:
    result = generate_response(
        user_input,
        st.session_state.history,
        prev_sentiment=st.session_state.last_sentiment,
        memory_context=st.session_state.memory_context
    )

    st.session_state.history = result["history"]
    st.session_state.last_sentiment = result["sentiment"]
    st.session_state.memory_context = result["memory_context"]

    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(result["reply"])

    if result.get("escalate"):
        st.warning("🚨 Escalation Triggered: Human agent required!")
        st.markdown("### 🧾 Handoff Summary for Agent")
        st.info(result["handoff_summary"])
