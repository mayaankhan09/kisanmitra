"""
KisanMitra — Phase 2: The Agent Core (minimal)
==============================================
A farmer asks a free-text question (Hindi or English). Gemini decides which
tools to call, calls get_weather + get_mandi_price automatically, then reasons
across both to give ONE clear recommendation in the farmer's language.

This is the heart of the project. It uses the google-genai SDK's AUTOMATIC
function calling: you just hand Gemini your Python functions as tools, and it
does the call-the-tool / read-the-result / decide loop for you.

Run order in a notebook:
  1. pip install google-genai pandas requests
  2. Make sure mandi_data.py is importable and you have a snapshot loaded.
  3. Paste your Gemini API key below.
  4. ask("मेरे पास धान है, अभी बेचूं या रुकूं?  जिला रायपुर")
"""

import os

from google import genai
from google.genai import types

# Your data tools from Phase 1
from mandi_data import get_mandi_price as _raw_mandi_price, load_store
import requests

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"   # fast, free-tier friendly, great for this

# Load the mandi store once so the tool is fast on every call
_STORE = load_store()

# A small map so the agent can turn a district name into coordinates for weather.
# Add more as you like; these cover Chhattisgarh's main areas.
DISTRICT_COORDS = {
    "raipur": (21.2514, 81.6296),
    "durg": (21.1904, 81.2849),
    "bilaspur": (22.0797, 82.1409),
    "raigarh": (21.8974, 83.3950),
    "jagdalpur": (19.0748, 82.0119),
    "ambikapur": (23.1206, 83.1959),
}

# ---------------------------------------------------------------------------
# TOOLS — these are what Gemini will call. The docstrings MATTER: Gemini reads
# them to decide when and how to call each tool. Keep them clear.
# ---------------------------------------------------------------------------

def get_weather_for_district(district: str) -> str:
    """Get the 7-day weather forecast for a district in Chhattisgarh, India.
    Use this to understand rainfall and temperature before advising a farmer
    about harvesting, spraying, or irrigation.

    Args:
        district: District name, e.g. 'Raipur', 'Durg', 'Bilaspur'.
    """
    key = district.strip().lower()
    lat, lon = DISTRICT_COORDS.get(key, DISTRICT_COORDS["raipur"])
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "timezone": "Asia/Kolkata", "forecast_days": 7,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        d = r.json()["daily"]
    except Exception as e:
        return f"Weather unavailable ({e})."
    lines = []
    for i in range(min(5, len(d["time"]))):
        lines.append(
            f"{d['time'][i]}: {d['temperature_2m_min'][i]}-{d['temperature_2m_max'][i]}°C, "
            f"rain {d['precipitation_sum'][i]}mm ({d['precipitation_probability_max'][i]}% chance)")
    rain3 = round(sum(d["precipitation_sum"][:3]), 1)
    return (f"Weather for {district} (next 5 days):\n" + "\n".join(lines) +
            f"\nTotal rain next 3 days: {rain3} mm.")


def get_crop_price(commodity: str, district: str = "Raipur") -> str:
    """Get recent mandi (market) prices for a crop, in Rupees per quintal.
    Use this to advise a farmer whether prices are good for selling.
    Returns min/max/modal prices and which market and date they are from.

    Args:
        commodity: Crop name, e.g. 'Paddy', 'Onion', 'Tomato', 'Wheat'.
        district: Farmer's district in Chhattisgarh, e.g. 'Raipur'.
    """
    res = _raw_mandi_price(commodity, state="Chhattisgarh",
                           district=district, df=_STORE, limit=8)
    # Drop obviously-bad rows (e.g. junk Rs 20 prices) for sanity
    good = [r for r in res["records"]
            if r.get("modal_price") and r["modal_price"] >= 100]
    if not good:
        return res["summary"]
    lines = [f"{r['market']} ({r['arrival_date']}): "
             f"modal Rs {int(r['modal_price'])}/qtl "
             f"[min {int(r['min_price'])}, max {int(r['max_price'])}]"
             for r in good[:6]]
    return f"{res['summary']}\nRecent records:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# THE AGENT — system prompt + one call with both tools attached
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are KisanMitra, a friendly farming advisor for small
farmers in Chhattisgarh, India.

Your job: given a farmer's question, give ONE clear, practical recommendation.

Rules:
- ALWAYS reason across BOTH weather AND market price before deciding. Call the
  weather tool and the price tool, then combine what they tell you.
- If the farmer asks whether to sell/harvest/wait, your answer must weigh:
  upcoming rain (can they harvest/dry the crop?) AND current price trend
  (is it worth selling now?).
- Reply in the SAME language the farmer used (Hindi or English). If they wrote
  in Hindi, answer in simple Hindi.
- Keep it short and actionable: 3-5 sentences. Start with the recommendation,
  then the 'because'. Mention the actual numbers you used (price, rain) so the
  farmer trusts the advice.
- If data is missing or from a nearby market, say so honestly.
- Never invent prices or weather. Only use what the tools return.
"""

_client = genai.Client(api_key=GEMINI_API_KEY)


def ask(question: str) -> str:
    """Ask KisanMitra a free-text farming question (Hindi or English)."""
    response = _client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[get_weather_for_district, get_crop_price],
            temperature=0.3,
        ),
    )
    return response.text


if __name__ == "__main__":
    # These will only run with a real key + internet for Gemini & weather.
    print(ask("I grow paddy in Raipur. Should I sell now or wait? "))
    print("\n" + "=" * 60 + "\n")
    print(ask("मेरे पास रायपुर में प्याज है, अभी बेचूं या रुकूं?"))
