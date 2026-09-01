import streamlit as st
from openai import OpenAI


# Show title and description.
st.title("Lab 2 Document ")

st.write(
    "Upload a document below and ask a question about it – GPT will answer! " 
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

options = ["Nano", "Mini", "Advanced Model"]
model = st.sidebar.radio("Models", options, index = None)

summary_type = st.sidebar.selectbox(
    'Summary Type',
    ('Summarize in 100 words',
     'Summarize in 2 connecting paragraphs',
     'Summarize in 5 bullet points'
)
)


if model == "Nano":
    model_choice = "gpt-5-nano"
elif model == "Mini":
    model_choice = "gpt-5-mini"
elif model == "Advanced Model":
    model_choice = "gpt-5"
else:
    model_choice = None



openai_api_key = st.secrets["My_newkey"]


# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md")
)

if uploaded_file and summary_type:

    # Process the uploaded file and summary type.
    document = uploaded_file.read().decode()
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {summary_type}",
        }
    ]

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model=model_choice,
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)
