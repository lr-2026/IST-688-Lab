import streamlit as st
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from openai import OpenAI
import chromadb   
from pathlib import Path
from PyPDF2 import PdfReader


# Create ChormaDB client
if "Lab4_VectorDB" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_Lab")
    collection = chroma_client.get_or_create_collection(name="Lab4Collection")
    if collection.count() == 0:
        load_pdfs_to_collection("./Lab-04-Data/", collection)
    st.session_state.Lab4_VectorDB = collection
else:
    collection = st.session_state.Lab4_VectorDB

#### USING CHROMA DB WITH OPENAI EMBEDDINGS ####

st.title("Question Answering Chatbot")

# Create the OpenAI client once and store it in session_state
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = st.session_state.client

# Model selector (mini vs regular)
openai_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
model_to_use = "gpt-4o-mini" if openai_model == "mini" else "gpt-4o"



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


    #### QUERYING A COLLECTION — ONLY USED FOR TESTING ####

topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

if topic:
    client = st.session_state.client  # <-- use your actual client variable name here

    response = client.embeddings.create(
        input=topic,
        model='text-embedding-3-small'
    )

    # Get the embedding
    query_embedding = response.data[0].embedding

    # Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # The number of closest documents to return
    )

    # Display the results
    st.subheader(f'Results for: {topic}')

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['ids'][0][i]

        st.write(f'**{i+1}. {doc_id}**')
else:
    st.info('Enter a topic in the sidebar to search the collection')    

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