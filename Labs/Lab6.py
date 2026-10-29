import os
from urllib.parse import quote
import requests
import streamlit as st
from langchain.agents import create_agent

# LangChain reads the key from this environment variable
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# ---------------------------------------------------------------
# Part B: weather tool (same function as Lab 5)
# ---------------------------------------------------------------
def get_weather_data(location: str = "Syracuse, NY") -> dict:
    """
    Fetch current weather conditions for a location from wttr.in.

    location can be a city, a zip code, an airport code ('SYR'),
    or a landmark ('Eiffel+Tower'). Defaults to 'Syracuse, NY'.

    Returns the values needed for clothing / activity advice:
    current temp, feels-like, description, wind, humidity, UV,
    today's high/low, max chance of rain, and how conditions
    change through the day (morning / afternoon / evening).
    """
    if not location:
        location = "Syracuse, NY"

    url = f"https://wttr.in/{quote(location)}?format=j1"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return {"error": f"wttr.in error: status {response.status_code}"}

    try:
        data = response.json()
    except ValueError:
        return {"error": f"Could not find a location named {location}"}

    current = data["current_condition"][0]
    today = data["weather"][0]
    hourly = today["hourly"]

    chance_of_rain = max(int(h.get("chanceofrain", 0)) for h in hourly)

    def slot(time_code: str) -> dict:
        for h in hourly:
            if h["time"] == time_code:
                return {
                    "temp_F": float(h["tempF"]),
                    "description": h["weatherDesc"][0]["value"],
                    "chance_of_rain_pct": int(h.get("chanceofrain", 0)),
                }
        return {}

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
        "uv_index": float(current.get("uvIndex", 0)),
        "high_F": float(today["maxtempF"]),
        "low_F": float(today["mintempF"]),
        "chance_of_rain_pct": chance_of_rain,
        "through_the_day": {
            "morning_9am": slot("900"),
            "afternoon_3pm": slot("1500"),
            "evening_9pm": slot("2100"),
        },
    }


# ---------------------------------------------------------------
# Part C: the agent (replaces client, tools JSON, and both API calls)
# ---------------------------------------------------------------
agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather_data],
    system_prompt=(
        "You are a helpful assistant that gives clothing and outdoor "
        "activity advice based on the weather. Use the get_weather_data "
        "tool when you need weather data. If the user does not give a "
        "location, use 'Syracuse, NY'. Consider how the weather changes "
        "from morning to evening, rain, wind, and UV. Keep it short and "
        "use bullet points."
    ),
)

st.title("👕 What to Wear Bot (LangChain)")
st.write(
    "Enter a city and get clothing suggestions and outdoor activity ideas "
    "based on the current weather."
)

city = st.text_input("City", placeholder="e.g. Syracuse, NY")
go = st.button("Get advice")

if go:
    if city.strip():
        user_prompt = (
            f"What should I wear today in {city}, and what outdoor "
            f"activities would be appropriate?"
        )
    else:
        user_prompt = (
            "What should I wear today, and what outdoor activities "
            "would be appropriate?"
        )

    with st.spinner("Thinking..."):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]}
        )

    advice = result["messages"][-1].content

    with st.chat_message("assistant"):
       st.write(advice.replace("~", "\\~"))

    # -----------------------------------------------------------
    # Part D: look inside the agent's conversation
    # -----------------------------------------------------------
    with st.expander("Behind the scenes"):
        for message in result["messages"]:
            st.write(f"**{message.type}**")
            st.write(message.content)
            if message.type == "ai" and message.tool_calls:
                st.write(message.tool_calls)


# ---------------------------------------------------------------
# Part D answers (step 12)
# ---------------------------------------------------------------
# a. For one city there are 4 messages:
#    1. human - my question built from the city box
#    2. ai    - the model asking to call get_weather_data (content is empty)
#    3. tool  - the JSON returned by get_weather_data
#    4. ai    - the final clothing/activity advice
#    (With "Syracuse, NY and Lima, Peru" there were 5 messages: two tool
#     messages, because the model asked for both cities in one ai message.)
#
# b. Message 2 (the first ai message) asks for the tool. Its content is
#    empty; the request is in message.tool_calls (name, args, id).
#    The tool message has the same id, so each result is matched to its request.
#
# c. In Lab 5, my own code created each message:
#    1. human - lines 149-160: built the messages list (system + user prompt)
#    2. ai    - lines 164-169: first client.chat.completions.create(tools=tools),
#               then lines 177-193: appended the assistant tool_calls message
#    3. tool  - lines 195-212: for loop ran get_current_weather and appended
#               a {"role": "tool", "tool_call_id": ...} message for each call
#    4. ai    - lines 229-232: second client.chat.completions.create
#    Lab 5 also added an extra user message (lines 215-226) before the
#    second call. LangChain did not need that.