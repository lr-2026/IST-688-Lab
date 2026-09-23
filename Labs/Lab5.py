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
api_key = st.secrets.get("My_newkey", None)
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

