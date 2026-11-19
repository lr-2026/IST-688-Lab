import asyncio
import json
import time
from urllib.parse import quote

import requests
import streamlit as st
from openai import OpenAI

# Lab 11 Part A: the OpenAI Agents SDK (pip install openai-agents)
from agents import Agent, ItemHelpers, Runner, function_tool, set_default_openai_key

# ---------------------------------------------------------------
# Weather data function (from Lab 5)
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
api_key = None
for name in ("OPENAI_API_KEY", "My_newkey"):
    if name in st.secrets:
        api_key = st.secrets[name]
        break
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🔑")
    st.stop()

client = OpenAI(api_key=api_key)

# Lab 11 Part A: the Agents SDK keeps its own key, so give it the same one
set_default_openai_key(api_key)


# ---------------------------------------------------------------
# Lab 11 Part B: the same weather function, as a tool for agents
# ---------------------------------------------------------------
@function_tool
def get_weather(location: str = "Syracuse, NY") -> dict:
    """Get today's weather for a location: current temperature, feels-like,
    conditions, wind, humidity, UV, high/low, chance of rain, and how it
    changes from morning to evening. Use 'Syracuse, NY' if no location is given."""
    return get_current_weather(location)


# ---------------------------------------------------------------
# Lab 11 Part C: three specialists and a manager
# ---------------------------------------------------------------
MODEL = "gpt-4o-mini"   # same model as Lab 5 (the deck uses gpt-5-mini; either works)

# Specialist 1: the only agent with a tool
weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "Summarize today's weather for the location you are given. Facts only: "
        "temperature, feels-like, conditions, wind, UV, chance of rain, and how "
        "it changes from morning to evening."
    ),
    tools=[get_weather],
    model=MODEL,
)

# Specialist 2: no tools, it only reasons about the weather it is told
stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You are told today's weather. Recommend what to wear. Consider how the "
        "temperature changes through the day, rain, wind and UV. Be specific and short."
    ),
    model=MODEL,
)

# Specialist 3: no tools either
activities_agent = Agent(
    name="Activities agent",
    instructions=(
        "You are told today's weather. Suggest outdoor activities that suit it. "
        "Be specific and short."
    ),
    model=MODEL,
)

# The manager calls each specialist like a tool, then writes the one final answer
manager = Agent(
    name="Manager",
    handoff_description="Outfits and outdoor activities for today.",   # used by the Bonus triage
    instructions=(
        "You help people decide what to wear and what to do today. "
        "First call weather_report for the location (use 'Syracuse, NY' if none is given). "
        "Then call clothing_advice and activity_ideas, passing along the weather you got. "
        "Combine everything into one short answer with bullet points."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description="Summarize today's weather for a location.",
        ),
        stylist_agent.as_tool(
            tool_name="clothing_advice",
            tool_description="Recommend clothing for a described forecast.",
        ),
        activities_agent.as_tool(
            tool_name="activity_ideas",
            tool_description="Suggest outdoor activities for a described forecast.",
        ),
    ],
    model=MODEL,
)


# ---------------------------------------------------------------
# Lab 11 Part E: ONE agent that does everything (to compare with the team)
# ---------------------------------------------------------------
single_agent = Agent(
    name="Single agent",
    instructions=(
        "You help people decide what to wear and what to do today. "
        "Get the weather with get_weather (use 'Syracuse, NY' if no location is given). "
        "Then recommend clothing (consider how it changes from morning to evening, "
        "rain, wind and UV) and suggest outdoor activities. "
        "Keep it short and use bullet points."
    ),
    tools=[get_weather],
    model=MODEL,
)

# ---------------------------------------------------------------
# Lab 11 Bonus: triage with handoffs
# ---------------------------------------------------------------
packing_agent = Agent(
    name="Packing agent",
    handoff_description="Packing lists for trips.",
    instructions=(
        "You write short packing lists for trips. Get the weather for the "
        "destination with get_weather, then list what to pack. Use bullet points."
    ),
    tools=[get_weather],
    model=MODEL,
)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Trips and packing go to the Packing agent. "
        "Outfits and activities for today go to the Manager. "
        "Do not answer yourself, always hand off."
    ),
    handoffs=[packing_agent, manager],
    model=MODEL,
)


def run_agent(agent, prompt: str):
    # Runner.run_sync can complain about the event loop inside Streamlit,
    # so run the async version ourselves
    return asyncio.run(Runner.run(agent, prompt))


# ---------------------------------------------------------------
# Lab 11 Part D: look behind the scenes
# ---------------------------------------------------------------
def show_behind_the_scenes(result, seconds: float):
    with st.expander("Behind the scenes", expanded=True):
        usage = result.context_wrapper.usage          # the cost of the run
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model requests", usage.requests)
        c2.metric("Total tokens", usage.total_tokens)
        c3.metric("Seconds", f"{seconds:.1f}")
        c4.metric("Last agent", result.last_agent.name)  # who finished the job

        st.markdown("**Every step, in order** (`result.new_items`):")
        for i, item in enumerate(result.new_items, 1):
            raw = item.raw_item
            if item.type == "tool_call_item":
                detail = f"`{getattr(raw, 'name', '?')}` with {getattr(raw, 'arguments', '')}"
            elif item.type == "tool_call_output_item":
                detail = str(item.output)[:300]
            elif item.type == "message_output_item":
                detail = ItemHelpers.text_message_output(item)[:300]
            else:
                detail = ""
            st.markdown(f"{i}. **{item.type}** {detail}")


# --- Tool definition for the OpenAI API (the old Lab 5 way, kept for now) ---
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


# ---------------------------------------------------------------
# Lab 11 Part C: the team version
# ---------------------------------------------------------------
st.divider()
st.header("👥 Team version (Lab 11)")
st.write("A Manager asks the Weather, Stylist and Activities agents, then writes one answer.")

team_city = st.text_input("City for the team", value="Syracuse, NY", key="team_city")
if st.button("Ask the team"):
    team_prompt = (
        f"What should I wear today in {team_city}, and what outdoor "
        f"activities would be appropriate?"
    )
    with st.spinner("The team is working..."):
        start = time.time()
        team_result = run_agent(manager, team_prompt)
        team_seconds = time.time() - start
    st.subheader("Team answer")
    st.write(team_result.final_output)
    show_behind_the_scenes(team_result, team_seconds)   # Part D


# ---------------------------------------------------------------
# Lab 11 Part E: is the team worth it?
# ---------------------------------------------------------------
st.divider()
st.header("⚖️ Part E: Is the team worth it?")
st.write("Runs the single agent and the team on the same cities, then compares cost and speed.")

cities_text = st.text_area(
    "Cities (one per line)", "Syracuse, NY\nParis, France\nPhoenix, AZ", key="cmp_cities"
)

if st.button("Run comparison"):
    cities = [c.strip() for c in cities_text.splitlines() if c.strip()]
    if not cities:
        st.warning("Add at least one city.")
    else:
        rows, answers = [], []
        progress = st.progress(0.0, text="Running...")
        for i, c in enumerate(cities):
            prompt = (
                f"What should I wear today in {c}, and what outdoor "
                f"activities would be appropriate?"
            )
            for label, agent in (("Single agent", single_agent), ("Team", manager)):
                start = time.time()
                res = run_agent(agent, prompt)
                secs = time.time() - start
                u = res.context_wrapper.usage
                rows.append(
                    {
                        "City": c,
                        "Version": label,
                        "Model requests": u.requests,
                        "Total tokens": u.total_tokens,
                        "Seconds": round(secs, 1),
                    }
                )
                answers.append((c, label, res.final_output))
            progress.progress((i + 1) / len(cities), text=f"Finished {c}")
        st.session_state.cmp_rows = rows
        st.session_state.cmp_answers = answers

if "cmp_rows" in st.session_state:
    rows = st.session_state.cmp_rows
    st.subheader("Per city")
    st.dataframe(rows)

    st.subheader("Average per run")
    summary = []
    for label in ("Single agent", "Team"):
        r = [x for x in rows if x["Version"] == label]
        n = len(r)
        summary.append(
            {
                "Version": label,
                "Model requests": round(sum(x["Model requests"] for x in r) / n, 1),
                "Total tokens": round(sum(x["Total tokens"] for x in r) / n),
                "Seconds": round(sum(x["Seconds"] for x in r) / n, 1),
            }
        )
    st.table(summary)
    st.caption(
        "Numbers come from result.context_wrapper.usage. If the Team's requests look "
        "too low, the specialists' own calls may not be counted: check the Traces "
        "tab on the OpenAI dashboard."
    )

    st.subheader("Was the answer better? You judge")
    for city, label, text in st.session_state.cmp_answers:
        with st.expander(f"{city} - {label}"):
            st.write(text)


# ---------------------------------------------------------------
# Lab 11 Bonus: triage with handoffs
# ---------------------------------------------------------------
st.divider()
st.header("🔀 Bonus: Triage with handoffs")
st.write(
    "The Triage agent hands the whole conversation to a specialist, and the "
    "specialist answers you directly."
)

triage_q = st.text_input(
    "Ask something", "Packing for Lima, Peru?", key="triage_q",
    help="Try a trip question and a 'what should I wear today' question.",
)
if st.button("Ask triage"):
    with st.spinner("Triage is deciding..."):
        start = time.time()
        triage_result = run_agent(triage_agent, triage_q)
        triage_seconds = time.time() - start
    st.subheader(f"Answered by: {triage_result.last_agent.name}")
    st.write(triage_result.final_output)
    show_behind_the_scenes(triage_result, triage_seconds)
