import streamlit as st
import sys
import json
import requests
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Part A: weather data function
# ---------------------------------------------------------------------------

def get_current_weather(location: str = "Syracuse, NY") -> dict:
    """
    Fetch current weather conditions for a location from wttr.in.

    location can be a city, a zip code, an airport code ('SYR'),
    or a landmark ('Eiffel+Tower'). Defaults to 'Syracuse, NY' if
    nothing is provided.

    Returns a dict with the fields needed to give clothing / activity
    advice: temperature, feels-like temperature, a text description,
    wind speed, humidity, chance of rain (from today's hourly data),
    and UV index.
    """
    if not location:
        location = "Syracuse, NY"

    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f"Could not find a location named {location}")

    current = data["current_condition"][0]

    # today's hourly forecast, used to pull a max chance-of-rain value
    today_hourly = data["weather"][0]["hourly"]
    chance_of_rain = max(int(hour.get("chanceofrain", 0)) for hour in today_hourly)

    return {
        "location": location,
        "resolved_location": data.get("nearest_area", [{}])[0]
        .get("areaName", [{}])[0]
        .get("value", location),
        "temperature_F": float(current["temp_F"]),
        "feels_like_F": float(current["FeelsLikeF"]),
        "description": current["weatherDesc"][0]["value"],
        "wind_mph": float(current["windspeedMiles"]),
        "humidity_pct": float(current["humidity"]),
        "chance_of_rain_pct": chance_of_rain,
        "uv_index": float(current.get("uvIndex", 0)),
    }


# ---------------------------------------------------------------------------
# Part B: "What to Wear" bot (Streamlit + OpenAI)
# ---------------------------------------------------------------------------

st.title("👕 What to Wear Bot")
st.write(
    "Enter a city and get clothing suggestions and outdoor activity ideas "
    "based on the current weather."
)

# --- OpenAI client setup ---------------------------------------------------
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🔑")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Tool definition for the OpenAI API ------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get the current weather conditions for a given location, "
                "including temperature, feels-like temperature, conditions, "
                "wind, humidity, chance of rain, and UV index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state/country, e.g. 'Syracuse, NY' "
                            "or 'Paris, France'. Defaults to 'Syracuse, NY' "
                            "if not provided."
                        ),
                    }
                },
                "required": [],
            },
        },
    }
]

available_functions = {"get_current_weather": get_current_weather}

# --- UI ---------------------------------------------------------------------
city = st.text_input("City", placeholder="e.g. Syracuse, NY")
go = st.button("What should I wear?")

if go:
    user_prompt = (
        f"What should I wear today in {city}, and what outdoor activities "
        f"would be appropriate?"
        if city
        else "What should I wear today, and what outdoor activities would be appropriate?"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that gives clothing and outdoor "
                "activity advice based on current weather. Always use the "
                "get_current_weather tool to check conditions before giving "
                "advice. If the user does not specify a location, use "
                "'Syracuse, NY' as the default."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    with st.spinner("Checking the weather..."):
        # First call: let the model decide whether to invoke the tool
        first_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    response_message = first_response.choices[0].message
    tool_calls = response_message.tool_calls

    weather_data = None

    if tool_calls:
        # Append the assistant's tool-call message to the conversation.
        # Must be converted to a plain dict (not the raw SDK object) before
        # being sent back in the next messages list.
        messages.append(response_message.model_dump())

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments or "{}")
            function_to_call = available_functions[function_name]

            try:
                function_response = function_to_call(**function_args)
                weather_data = function_response
            except Exception as e:
                function_response = {"error": str(e)}

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )

        with st.spinner("Putting together your advice..."):
            # Second call: feed the tool result back for the actual advice
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )

        advice = second_response.choices[0].message.content
    else:
        # Model answered directly without needing the tool
        advice = response_message.content

    # --- Display results -----------------------------------------------
    if weather_data:
        st.subheader(f"Current weather in {weather_data['resolved_location']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{weather_data['temperature_F']:.0f}°F")
        col2.metric("Feels like", f"{weather_data['feels_like_F']:.0f}°F")
        col3.metric("Chance of rain", f"{weather_data['chance_of_rain_pct']:.0f}%")
        st.caption(
            f"{weather_data['description']} • "
            f"Wind {weather_data['wind_mph']:.0f} mph • "
            f"Humidity {weather_data['humidity_pct']:.0f}% • "
            f"UV index {weather_data['uv_index']:.0f}"
        )

    st.subheader("Recommendation")
    st.write(advice)

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


# Create the OpenAI client once and store it in session_state

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["My_newkey"])
client = st.session_state.client

# Model selector mini vs regular
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


# Part A: Building the ChromaDB vector database

if "Lab4_VectorDB" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_Lab")
    collection = chroma_client.get_or_create_collection(name="Lab4Collection")
    if collection.count() == 0:
        load_pdfs_to_collection("./Lab-04-Data/", collection)
    st.session_state.Lab4_VectorDB = collection
else:
    collection = st.session_state.Lab4_VectorDB


# Part B: The actual RAG chatbot

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

    #  RAG step: retrieve relevant syllabus context for this prompt 
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
