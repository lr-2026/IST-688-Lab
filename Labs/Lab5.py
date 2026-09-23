import json
from urllib.parse import quote

import requests
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------
# Part A: weather data function
# ---------------------------------------------------------------


def get_current_weather(location: str = "Syracuse, NY") -> dict:
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

    # quote() makes spaces and commas safe for the URL
    url = f"https://wttr.in/{quote(location)}?format=j1"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f"Could not find a location named {location}")

    current = data["current_condition"][0]
    today = data["weather"][0]
    hourly = today["hourly"]  # 8 entries: 0, 300, 600, ... 2100

    chance_of_rain = max(int(h.get("chanceofrain", 0)) for h in hourly)

    def slot(time_code: str) -> dict:
        """Pick one hourly entry, e.g. '900' = 9 AM."""
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
# Part B: "What to Wear" bot (Streamlit + OpenAI)
# ---------------------------------------------------------------

st.title("👕 What to Wear Bot")
st.write(
    "Enter a city and get clothing suggestions and outdoor activity ideas "
    "based on the current weather."
)

# --- OpenAI client setup ---
# The name must match the key in your .streamlit/secrets.toml
api_key = st.secrets.get("My_newkey", None)
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🔑")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Tool definition for the OpenAI API ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get current weather and today's forecast for a location: "
                "temperature, feels-like, conditions, wind, humidity, UV, "
                "high/low, chance of rain, and morning/afternoon/evening changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state/country, e.g. 'Syracuse, NY' "
                            "or 'Paris, France'. Use 'Syracuse, NY' if the "
                            "user gives no location."
                        ),
                    }
                },
                "required": [],
            },
        },
    }
]

available_functions = {"get_current_weather": get_current_weather}

# --- UI (not a chatbot: one input, one answer) ---
city = st.text_input("City", placeholder="e.g. Syracuse, NY")
go = st.button("What should I wear?")

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

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that gives clothing and outdoor "
                "activity advice based on the weather. Use the "
                "get_current_weather tool when you need weather data. If the "
                "user does not give a location, use 'Syracuse, NY'."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    # First call: the model decides whether it needs the tool
    with st.spinner("Checking the weather..."):
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
        # Add the assistant's tool-call message as a plain dict
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

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
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_response),
                }
            )

        # Second call: weather info is in the prompt, ask for the advice
        messages.append(
            {
                "role": "user",
                "content": (
                    "Using the weather information above, suggest appropriate "
                    "clothes to wear today (consider how it changes from "
                    "morning to evening, rain, wind, and UV) and suggest "
                    "outdoor activities that suit the weather. Keep it short "
                    "and use bullet points."
                ),
            }
        )

        with st.spinner("Putting together your advice..."):
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )

        advice = second_response.choices[0].message.content
    else:
        # Model answered without needing the tool
        advice = response_message.content

    # --- Display results ---
    if weather_data:
        st.subheader(f"Current weather in {weather_data['resolved_location']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{weather_data['temperature_F']:.0f}°F")
        col2.metric("Feels like", f"{weather_data['feels_like_F']:.0f}°F")
        col3.metric(
            "High / Low",
            f"{weather_data['high_F']:.0f}° / {weather_data['low_F']:.0f}°",
        )
        col4.metric("Chance of rain", f"{weather_data['chance_of_rain_pct']:.0f}%")
        st.caption(
            f"{weather_data['description']} • "
            f"Wind {weather_data['wind_mph']:.0f} mph • "
            f"Humidity {weather_data['humidity_pct']:.0f}% • "
            f"UV index {weather_data['uv_index']:.0f}"
        )

    st.subheader("Recommendation")
    st.write(advice)
