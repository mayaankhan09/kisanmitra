"""
KisanMitra — Phase 3: Streamlit web app
=======================================
Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub -> Streamlit Community Cloud (free public URL)

Keys are read from Streamlit "secrets" (never hard-coded), so this file is
safe to put on a public GitHub repo.
  - Locally: create  .streamlit/secrets.toml  with your keys (see secrets.toml.example)
  - On Streamlit Cloud: paste the same keys in the app's Settings -> Secrets box
"""

import streamlit as st
import requests
import pandas as pd
from google import genai
from google.genai import types

from mandi_data import get_mandi_price as _raw_mandi_price, load_store

# ---------------------------------------------------------------------------
# Config & keys (from secrets)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"

DISTRICT_COORDS = {
    "Raipur": (21.2514, 81.6296),
    "Durg": (21.1904, 81.2849),
    "Bilaspur": (22.0797, 82.1409),
    "Raigarh": (21.8974, 83.3950),
    "Jagdalpur": (19.0748, 82.0119),
    "Ambikapur": (23.1206, 83.1959),
}
CROPS = ["Paddy", "Rice", "Wheat", "Onion", "Tomato", "Potato", "Maize"]

# ---------------------------------------------------------------------------
# Load data + client once (cached so the app is fast)
# ---------------------------------------------------------------------------
@st.cache_data
def _get_store():
    return load_store()

_STORE = _get_store()

@st.cache_resource
def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Tools (same logic as the notebook agent) — return clean text for Gemini
# and ALSO expose raw data so the UI can show it.
# ---------------------------------------------------------------------------
def _weather_raw(district: str):
    lat, lon = DISTRICT_COORDS.get(district, DISTRICT_COORDS["Raipur"])
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "timezone": "Asia/Kolkata", "forecast_days": 7,
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    return r.json()["daily"]

def get_weather_for_district(district: str) -> str:
    """Get the 7-day weather forecast for a district in Chhattisgarh, India.
    Use this to understand rainfall and temperature before advising about
    harvesting, spraying, or irrigation.
    Args:
        district: District name, e.g. 'Raipur', 'Durg', 'Bilaspur'.
    """
    try:
        d = _weather_raw(district)
    except Exception as e:
        return f"Weather unavailable ({e})."
    lines = [f"{d['time'][i]}: {d['temperature_2m_min'][i]}-{d['temperature_2m_max'][i]}°C, "
             f"rain {d['precipitation_sum'][i]}mm ({d['precipitation_probability_max'][i]}%)"
             for i in range(min(5, len(d["time"])))]
    rain3 = round(sum(d["precipitation_sum"][:3]), 1)
    return (f"Weather for {district} (5 days):\n" + "\n".join(lines) +
            f"\nTotal rain next 3 days: {rain3} mm.")

def get_crop_price(commodity: str, district: str = "Raipur") -> str:
    """Get recent mandi (market) prices for a crop, in Rupees per quintal.
    Use this to advise whether prices are good for selling. Returns min/max/modal
    prices with market and date.
    Args:
        commodity: Crop name, e.g. 'Paddy', 'Onion', 'Tomato', 'Wheat'.
        district: Farmer's district in Chhattisgarh, e.g. 'Raipur'.
    """
    res = _raw_mandi_price(commodity, state="Chhattisgarh", district=district,
                           df=_STORE, limit=8)
    good = [r for r in res["records"]
            if r.get("modal_price") and r["modal_price"] >= 100]
    if not good:
        return res["summary"]
    lines = [f"{r['market']} ({r['arrival_date']}): modal Rs {int(r['modal_price'])}/qtl "
             f"[{int(r['min_price'])}-{int(r['max_price'])}]" for r in good[:6]]
    return f"{res['summary']}\n" + "\n".join(lines)

SYSTEM_PROMPT = """You are KisanMitra, a friendly farming advisor for small
farmers in Chhattisgarh, India. Given a farmer's question, give ONE clear,
practical recommendation.
Rules:
- ALWAYS reason across BOTH weather AND market price. Call both tools, then combine them.
- For sell/harvest/wait questions, weigh upcoming rain AND the current price.
- Reply in the SAME language the farmer used (Hindi or English). Simple words.
- 3-5 sentences. Start with the recommendation, then the 'because', citing the
  actual price and rain numbers so the farmer trusts it.
- If data is missing or from a nearby market, say so honestly. Never invent data.
"""

def ask_agent(question: str) -> str:
    client = _get_client()
    resp = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[get_weather_for_district, get_crop_price],
            temperature=0.3,
        ),
    )
    return resp.text

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="KisanMitra — किसान मित्र", page_icon="🌾", layout="centered")

st.markdown(
    "<h1 style='margin-bottom:0'>🌾 KisanMitra</h1>"
    "<p style='color:#4b7d2e;font-size:1.1rem;margin-top:4px'>किसान मित्र — your AI farming advisor</p>",
    unsafe_allow_html=True,
)
st.caption("Ask in Hindi or English. The agent checks live weather + mandi prices, then advises.")

if not GEMINI_API_KEY:
    st.error("No Gemini API key found. Add GEMINI_API_KEY to Streamlit secrets.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    crop = st.selectbox("Crop / फसल", CROPS, index=0)
with col2:
    district = st.selectbox("District / जिला", list(DISTRICT_COORDS.keys()), index=0)

default_q = f"I grow {crop.lower()} in {district}. Should I sell now or wait?"
question = st.text_area("Your question / आपका सवाल", value=default_q, height=80)

if st.button("Ask KisanMitra / पूछें", type="primary"):
    with st.spinner("Checking weather and mandi prices…"):
        try:
            answer = ask_agent(question)
            st.success(answer)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # Transparency panel — show the raw data the agent reasoned over
    with st.expander("📊 Data the agent used (live)"):
        st.markdown("**Weather**")
        st.text(get_weather_for_district(district))
        st.markdown("**Mandi prices**")
        st.text(get_crop_price(crop, district))

st.markdown("---")
st.caption("Data: Open-Meteo (weather) + data.gov.in AGMARKNET (mandi prices). "
           "Prices may be from nearby markets when local data is unavailable.")
