import streamlit as st
from openai import OpenAI

st.title("Question Answering Chatbot")

# Model selector (mini vs regular)
openai_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
model_to_use = "gpt-4o-mini" if openai_model == "mini" else "gpt-4o"

# Create the OpenAI client once and store it in session_state
if "client" not in st.session_state:
    api_key = st.secrets["My_newkey"]
    st.session_state.client = OpenAI(api_key=api_key)

    # --- System prompt: defines the bot's behavior for Part C ---
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful assistant. Follow this exact behavior:\n"
        "1. When the user asks a question, answer it.\n"
        "2. After answering, always ask: 'Do you want more info?'\n"
        "3. If the user says yes (or similar), give more information on the "
        "same topic, then ask again: 'Do you want more info?'\n"
        "4. If the user says no (or similar), respond by asking what else "
        "you can help with.\n"
        "5. Always explain answers simply enough that a 10-year-old could "
        "understand them — use short sentences, simple words, and avoid "
        "jargon."
    ),
}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display existing chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# React to new user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client

    buffer_size = 4
    recent_messages = st.session_state.messages[-buffer_size:]
    messages_to_send = [SYSTEM_PROMPT] + recent_messages

    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages_to_send,
        stream=True,
    )
    
    

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})