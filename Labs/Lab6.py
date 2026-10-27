import os
from urllib.parse import quote
import requests
import streamlit as st
from langchain.agents import create_agent

os.environ["OPENAI_API_KEY"] = st.secrets["My_newkey"]


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


# --- Agent (replaces the client, tools JSON, and both API calls) ---
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

# --- UI ---
st.title("👕 What to Wear Bot (LangChain)")
location = st.text_input("City", placeholder="e.g. Syracuse, NY")

if st.button("Get advice"):
    with st.spinner("Thinking..."):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"What should I wear in {location}?"}]}
        )

    with st.chat_message("assistant"):
        st.write(result["messages"][-1].content)