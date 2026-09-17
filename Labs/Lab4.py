import streamlit as st
import sys

# Fix for ChromaDB's sqlite3 requirement on Streamlit Community Cloud.
# Wrapped in try/except so it also works locally, where pysqlite3-binary
# usually isn't installed (and isn't needed, since local sqlite3 is fine).
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # pysqlite3 not installed locally — fine, system sqlite3 works

from openai import OpenAI
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

st.title("Lab 4: Chatbot (RAG)")

# ---------------------------------------------------------------------------
# Create the OpenAI client once and store it in session_state
# ---------------------------------------------------------------------------
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = st.session_state.client

# Model selector (mini vs regular)
openai_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
model_to_use = "gpt-4o-mini" if openai_model == "mini" else "gpt-4o"


# ---------------------------------------------------------------------------
# Helper functions — must be defined BEFORE they're used below
# ---------------------------------------------------------------------------
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def add_to_collection(collection, text, file_name):
    embedding = get_embedding(text)
    collection.add(documents=[text], ids=[file_name], embeddings=[embedding])


def load_pdfs_to_collection(folder_path, collection):
    for pdf_file in sorted(Path(folder_path).glob("*.pdf")):
        text = extract_text_from_pdf(str(pdf_file))
        add_to_collection(collection, text, pdf_file.name)


def get_relevant_context(collection, query_text, n_results=3):
    """Embed the query, search the collection, and return the combined
    text of the top matches plus the filenames used (for transparency)."""
    query_embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    docs = results["documents"][0]
    ids = results["ids"][0]
    context_text = "\n\n".join(
        f"[Source: {doc_id}]\n{doc}" for doc_id, doc in zip(ids, docs)
    )
    return context_text, ids


# ---------------------------------------------------------------------------
# Part A: Build (or reuse) the ChromaDB vector database
# This runs AFTER the functions above are defined, so the call is safe.
# ---------------------------------------------------------------------------
if "Lab4_VectorDB" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_Lab")
    collection = chroma_client.get_or_create_collection(name="Lab4Collection")
    if collection.count() == 0:
        load_pdfs_to_collection("./Lab-04-Data/", collection)
    st.session_state.Lab4_VectorDB = collection
else:
    collection = st.session_state.Lab4_VectorDB


# ---------------------------------------------------------------------------
# Part A: Optional test search — kept behind a checkbox so it doesn't
# interfere with the real chatbot below. Check it to validate the vectorDB.
# ---------------------------------------------------------------------------
with st.sidebar:
    show_test_search = st.checkbox("Show Part A test search")

if show_test_search:
    topic = st.sidebar.text_input(
        "Topic", placeholder="Type your topic (e.g., GenAI)..."
    )
    if topic:
        context_text, ids = get_relevant_context(collection, topic, n_results=3)
        st.subheader(f"Results for: {topic}")
        for i, doc_id in enumerate(ids):
            st.write(f"**{i + 1}. {doc_id}**")
    else:
        st.info("Enter a topic in the sidebar to search the collection")


# ---------------------------------------------------------------------------
# Part B: The actual RAG chatbot
# ---------------------------------------------------------------------------
BASE_SYSTEM_PROMPT = (
    "You are a helpful course information assistant. You answer questions "
    "about course syllabi using the context provided below, which was "
    "retrieved from a vector database of syllabus PDFs.\n\n"
    "IMPORTANT: If you use information from the provided context to answer, "
    "explicitly say so at the start of your answer, e.g. 'Based on the "
    "course syllabus information I found...'. If the context doesn't "
    "contain relevant information, say so and answer from general "
    "knowledge instead, making clear you are not using the syllabus data.\n\n"
    "Context from syllabi:\n{context}"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you with course info?"}
    ]

# Display existing chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# React to new user input
if prompt := st.chat_input("Ask about a course..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- RAG step: retrieve relevant syllabus context for this prompt ---
    context_text, source_ids = get_relevant_context(collection, prompt, n_results=3)
    system_prompt = {
        "role": "system",
        "content": BASE_SYSTEM_PROMPT.format(context=context_text),
    }

    buffer_size = 4
    recent_messages = st.session_state.messages[-buffer_size:]
    messages_to_send = [system_prompt] + recent_messages

    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages_to_send,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
        with st.expander("Sources used for this answer"):
            st.write(", ".join(source_ids))

    st.session_state.messages.append({"role": "assistant", "content": response})
