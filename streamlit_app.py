"""
KisanMitra — Streamlit web app (redesigned UI, all-India)
===========================================================
Run locally:   streamlit run streamlit_app_v2.py
Deploy:        push to GitHub -> Streamlit Community Cloud

Keys are read from Streamlit secrets (never hard-coded).
  - Locally: .streamlit/secrets.toml  with GEMINI_API_KEY = "..."
  - On Streamlit Cloud: Settings -> Secrets
"""

import html
import json
import streamlit as st
import requests
import pandas as pd
from google import genai
from google.genai import types

from mandi_data import get_mandi_price as _raw_mandi_price, load_store

# ---------------------------------------------------------------------------
# Config & keys
# ---------------------------------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash-lite"   # generous free-tier limits

RAIPUR_FALLBACK = (21.2514, 81.6296)

STATES = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Keralam", "Madhya Pradesh",
    "Maharashtra", "Odisha", "Punjab", "Rajasthan", "Tamil Nadu", "Telangana",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
]
CROPS = [
    "Paddy", "Rice", "Wheat", "Maize", "Soyabean", "Cotton", "Groundnut",
    "Gram", "Onion", "Tomato", "Potato", "Chilli", "Brinjal", "Cauliflower",
    "Banana", "Mango",
]

# ---------------------------------------------------------------------------
# Geocoding + data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def geocode_place(place: str, state: str = ""):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 5, "country": "IN", "language": "en"},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception:
        return RAIPUR_FALLBACK
    if not results:
        return RAIPUR_FALLBACK
    if state:
        for res in results:
            if res.get("admin1", "").lower() == state.lower():
                return res["latitude"], res["longitude"]
    top = results[0]
    return top["latitude"], top["longitude"]

@st.cache_data
def _get_store():
    return load_store()

_STORE = _get_store()

@st.cache_resource
def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Raw data helpers (also used by the UI cards)
# ---------------------------------------------------------------------------
def _weather_raw(district: str, state: str):
    lat, lon = geocode_place(district, state)
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "timezone": "Asia/Kolkata", "forecast_days": 7,
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    return r.json()["daily"]

def weather_summary(district: str, state: str):
    """Return (text_for_agent, structured_dict_for_ui)."""
    try:
        d = _weather_raw(district, state)
    except Exception as e:
        return f"Weather unavailable ({e}).", None
    days = []
    for i in range(min(5, len(d["time"]))):
        days.append({
            "date": d["time"][i],
            "tmin": d["temperature_2m_min"][i],
            "tmax": d["temperature_2m_max"][i],
            "rain": d["precipitation_sum"][i],
            "rain_pct": d["precipitation_probability_max"][i],
        })
    rain3 = round(sum(x["rain"] for x in days[:3]), 1)
    text = (f"Weather for {district} (5 days):\n" +
            "\n".join(f"{x['date']}: {x['tmin']}-{x['tmax']}°C, rain {x['rain']}mm ({x['rain_pct']}%)"
                      for x in days) +
            f"\nTotal rain next 3 days: {rain3} mm.")
    return text, {"days": days, "rain3": rain3}

def price_summary(commodity: str, state: str, district: str):
    """Return (text_for_agent, structured_dict_for_ui)."""
    res = _raw_mandi_price(commodity, state=state, district=district, df=_STORE, limit=8)
    good = [r for r in res["records"]
            if r.get("modal_price") and r["modal_price"] >= 100]
    if not good:
        return res["summary"], {"summary": res["summary"], "rows": []}
    rows = [{
        "market": r["market"], "date": r["arrival_date"],
        "modal": int(r["modal_price"]), "lo": int(r["min_price"]), "hi": int(r["max_price"]),
    } for r in good[:6]]
    text = res["summary"] + "\n" + "\n".join(
        f"{r['market']} ({r['date']}): modal Rs {r['modal']}/qtl [{r['lo']}-{r['hi']}]" for r in rows)
    return text, {"summary": res["summary"], "rows": rows}

# ---------------------------------------------------------------------------
# Agent tools (wrap the helpers, read state from session)
# ---------------------------------------------------------------------------
def get_weather_for_district(district: str) -> str:
    """Get the 7-day weather forecast for a district in India.
    Use this to understand rainfall and temperature before advising about
    harvesting, spraying, or irrigation.
    Args:
        district: District name, e.g. 'Raipur', 'Pune', 'Nagpur'.
    """
    state = st.session_state.get("sel_state", "Chhattisgarh")
    text, _ = weather_summary(district, state)
    return text

def get_crop_price(commodity: str, district: str = "Raipur") -> str:
    """Get recent mandi (market) prices for a crop, in Rupees per quintal.
    Use this to advise whether prices are good for selling.
    Args:
        commodity: Crop name, e.g. 'Paddy', 'Onion', 'Tomato', 'Wheat'.
        district: Farmer's district, e.g. 'Raipur'.
    """
    state = st.session_state.get("sel_state", "Chhattisgarh")
    text, _ = price_summary(commodity, state, district)
    return text

# ---------------------------------------------------------------------------
# Language toggle helper (UI-driven; only steers the model's OUTPUT language)
# ---------------------------------------------------------------------------
_LANGUAGE_NAMES = {"English": "English", "हिंदी": "Hindi"}

def _forced_language():
    """Full language name to force in prompts, or None to fall back to the
    model's own 'reply in the farmer's language' behaviour."""
    return _LANGUAGE_NAMES.get(st.session_state.get("out_lang"))

def _is_hindi() -> bool:
    return st.session_state.get("out_lang") == "हिंदी"

def _pick(en_val, hi_val):
    """Pick the display value matching the current output-language toggle."""
    return hi_val if _is_hindi() else en_val

# ---------------------------------------------------------------------------
# Agent + structured decision
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are KisanMitra, a friendly farming advisor for small
farmers across India. Given a farmer's question, give ONE clear, practical
recommendation.
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
    system_instruction = SYSTEM_PROMPT
    forced = _forced_language()
    if forced:
        system_instruction += f"\nRespond entirely in {forced}."
    resp = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[get_weather_for_district, get_crop_price],
            temperature=0.3,
        ),
    )
    return resp.text

def classify_decision(answer: str, question: str) -> dict:
    """Ask the model to distill its own answer into a structured verdict for the
    decision card: verdict, bilingual headline + one-line summary, a confidence
    percentage, and 2-3 short bilingual reasoning bullets ("why this advice?").
    This is the ONE extra model call used to drive the redesigned UI cards."""
    client = _get_client()
    prompt = (
        "From this farming advice, distill a structured verdict for a UI card.\n"
        f"QUESTION: {question}\nADVICE: {answer}\n\n"
        "Return ONLY JSON, no markdown, with keys:\n"
        '{"verdict": "SELL" or "WAIT" or "OTHER", '
        '"headline_en": "<3-4 word action, e.g. Sell Now / Wait Before Selling>", '
        '"headline_hi": "<same in simple Hindi, e.g. अभी बेचें / कुछ दिन रुकें>", '
        '"summary_en": "<one plain sentence explaining the headline>", '
        '"summary_hi": "<same one sentence in simple Hindi>", '
        '"confidence_pct": <integer 0-100, how confident this recommendation is>, '
        '"reasoning_en": ["<short reason 1>", "<short reason 2>", "<short reason 3, optional>"], '
        '"reasoning_hi": ["<same reasons in simple Hindi>"]}'
    )
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        txt = resp.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(txt)
        if data.get("verdict") not in ("SELL", "WAIT", "OTHER"):
            data["verdict"] = "OTHER"
        try:
            data["confidence_pct"] = max(0, min(100, int(data.get("confidence_pct", 70))))
        except (TypeError, ValueError):
            data["confidence_pct"] = 70
        data.setdefault("summary_en", "")
        data.setdefault("summary_hi", "")
        data["reasoning_en"] = list(data.get("reasoning_en") or [])[:3]
        data["reasoning_hi"] = list(data.get("reasoning_hi") or [])[:3]
        return data
    except Exception:
        return {
            "verdict": "OTHER", "headline_en": "See advice", "headline_hi": "सलाह देखें",
            "summary_en": "Read the full advice below for details.",
            "summary_hi": "विवरण के लिए नीचे पूरी सलाह पढ़ें।",
            "confidence_pct": 60,
            "reasoning_en": ["Based on the latest weather and mandi data available."],
            "reasoning_hi": ["नवीनतम मौसम और मंडी डेटा के आधार पर।"],
        }

# ---------------------------------------------------------------------------
# Crop-doctor (multimodal)
# ---------------------------------------------------------------------------
DIAGNOSIS_PROMPT = """You are KisanMitra's crop-doctor. A farmer in India has
uploaded a photo of their crop. Look carefully at the leaf/plant.

Return ONLY JSON, no markdown, with keys:
{"crop": "<crop name, or 'Unclear' if not identifiable>",
 "disease": "<disease, pest, or deficiency name>",
 "confidence_pct": <integer 0-100, how confident this diagnosis is>,
 "what_to_do": ["<short, practical step 1>", "<step 2>", "<step 3, optional>"],
 "precautions": ["<short safety note, e.g. confirm with local Krishi Vigyan Kendra before spraying chemicals>"]}

Rules:
- If the image is NOT a plant/crop, set "disease" to a polite message asking
  for a clear crop photo, set "confidence_pct" to 0, and leave "what_to_do"
  empty.
- Prefer low-cost, locally available remedies first.
- Never recommend dangerous pesticide doses. Keep advice safe and general.
- Be honest about confidence_pct — a wrong confident diagnosis can hurt a farmer.
- Keep every string short and in simple, plain language a farmer can follow.
"""

def diagnose_image(image_bytes: bytes, mime_type: str) -> dict:
    client = _get_client()
    prompt = DIAGNOSIS_PROMPT
    forced = _forced_language()
    if forced:
        prompt += f"\nRespond entirely in {forced}."
    else:
        prompt += "\nRespond in simple Hindi and English mixed, the way a friendly local advisor would."
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    txt = resp.text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(txt)
    except Exception:
        data = {"crop": "Unclear", "disease": resp.text.strip(),
                 "confidence_pct": 0, "what_to_do": [], "precautions": []}
    try:
        data["confidence_pct"] = max(0, min(100, int(data.get("confidence_pct") or 0)))
    except (TypeError, ValueError):
        data["confidence_pct"] = 0
    data.setdefault("crop", "Unclear")
    data.setdefault("disease", "Unable to diagnose")
    data["what_to_do"] = list(data.get("what_to_do") or [])
    data["precautions"] = list(data.get("precautions") or [])
    return data

# ===========================================================================
# UI
# ===========================================================================
st.set_page_config(page_title="KisanMitra — AI Farming Assistant", page_icon="🌱", layout="wide")

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

THEME = st.session_state["theme"]

# ---- Styles: modern, minimal, soft cards, light/dark via CSS variables ----
LIGHT_VARS = """
  --km-bg:#f8faf8; --km-card-bg:#ffffff; --km-card-border:#e6ebe3;
  --km-text:#3c4a3a; --km-heading:#16321c; --km-text-muted:#73826d;
  --km-shadow:rgba(31,61,26,0.08); --km-input-bg:#ffffff;
"""
DARK_VARS = """
  --km-bg:#10160f; --km-card-bg:#17201a; --km-card-border:#283228;
  --km-text:#cfd9cc; --km-heading:#eef3ec; --km-text-muted:#8a988a;
  --km-shadow:rgba(0,0,0,0.55); --km-input-bg:#1b261e;
"""

BASE_CSS = """
<style>
:root { __VARS__ }

.stApp { background: var(--km-bg); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.1rem; max-width: 1220px; }
.stApp, .stApp p, .stApp span, .stApp label, .stApp li { color: var(--km-text); }
.stApp h1, .stApp h2, .stApp h3 { color: var(--km-heading); }

/* ---- top bar ---- */
.km-brand-row { display:flex; align-items:center; gap:10px; }
.km-logo-badge {
  width:40px; height:40px; border-radius:12px; flex-shrink:0;
  background:linear-gradient(135deg,#22b35a,#16a34a);
  display:flex; align-items:center; justify-content:center; font-size:1.25rem;
  box-shadow:0 4px 12px rgba(22,163,74,0.35);
}
.km-brand-name { font-weight:800; font-size:1.25rem; color:var(--km-heading); line-height:1.1; }
.km-brand-sub { font-size:0.78rem; color:var(--km-text-muted); margin-top:1px; }

/* ---- segmented / pill controls (nav + language) ---- */
div[data-testid="stButtonGroup"] {
  background: var(--km-card-bg); border:1px solid var(--km-card-border);
  border-radius:999px; padding:3px;
}
div[data-testid="stButtonGroup"] [aria-checked="false"] { color: var(--km-text-muted) !important; }
div[data-testid="stButtonGroup"] [aria-checked="true"] {
  background:#16a34a !important; color:#fff !important; border-radius:999px !important;
}
div[data-testid="stButtonGroup"] label, div[data-testid="stButtonGroup"] p { font-weight:600 !important; }

/* ---- buttons ---- */
button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg,#22b35a,#16a34a) !important;
  border:none !important; border-radius:12px !important; color:#fff !important;
  font-weight:600 !important; box-shadow:0 6px 16px rgba(22,163,74,0.25) !important;
}
button[data-testid="stBaseButton-secondary"] {
  background: var(--km-card-bg) !important; border:1px solid var(--km-card-border) !important;
  color: var(--km-text) !important; border-radius:12px !important; font-weight:600 !important;
}

/* ---- inputs ---- */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
  background: var(--km-input-bg) !important; color: var(--km-text) !important;
  border-radius:10px !important; border:1px solid var(--km-card-border) !important;
}
div[data-testid="stFileUploaderDropzone"] {
  background: var(--km-input-bg) !important; border:2px dashed var(--km-card-border) !important;
  border-radius:16px !important;
}

/* ---- expander ---- */
div[data-testid="stExpander"] {
  background: var(--km-card-bg); border:1px solid var(--km-card-border) !important;
  border-radius:16px !important;
}

/* ---- progress / confidence bar ---- */
div[data-testid="stProgressBarTrack"] { background: var(--km-card-border) !important; border-radius:999px !important; }
div[data-testid="stProgressBarTrack"] > div { background:#16a34a !important; border-radius:999px !important; }

/* ---- generic card ---- */
.km-card {
  background: var(--km-card-bg); border:1px solid var(--km-card-border); border-radius:16px;
  padding:18px 20px; box-shadow:0 6px 20px var(--km-shadow); margin-bottom:16px; height:100%;
}
.km-card-head { display:flex; align-items:center; gap:8px; font-size:0.82rem; font-weight:700;
  text-transform:uppercase; letter-spacing:0.06em; color:var(--km-text-muted); margin:0 0 10px; }
.km-big { font-size:1.7rem; font-weight:800; color:var(--km-heading); margin:0; }
.km-small { color:var(--km-text-muted); font-size:0.88rem; margin:4px 0 0; }

/* ---- recommendation card ---- */
.km-rec-card {
  background: var(--km-card-bg); border:1px solid var(--km-card-border); border-radius:16px;
  padding:20px 22px; box-shadow:0 6px 20px var(--km-shadow); height:100%;
}
.km-rec-label { font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--km-text-muted); margin:0 0 6px; }
.km-rec-headline { font-size:1.7rem; font-weight:800; margin:0 0 6px; line-height:1.15; }
.km-rec-summary { color:var(--km-text); font-size:0.95rem; margin:0 0 14px; }
.km-conf-row { display:flex; justify-content:space-between; font-size:0.82rem;
  color:var(--km-text-muted); margin-bottom:4px; }

/* ---- bullets ---- */
.km-bullet { display:flex; align-items:flex-start; gap:8px; font-size:0.92rem; margin:6px 0; color:var(--km-text); }
.km-bullet-icon { flex-shrink:0; }

/* ---- live data rows ---- */
.km-row { display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid var(--km-card-border);
  font-size:0.88rem; color:var(--km-text); }
.km-row:last-child { border-bottom:none; }

/* ---- forecast day chip ---- */
.km-day { background: var(--km-bg); border:1px solid var(--km-card-border); border-radius:12px;
  padding:10px 6px; text-align:center; }
.km-day-name { font-size:0.74rem; color:var(--km-text-muted); margin:0 0 4px; }
.km-day-icon { font-size:1.3rem; margin:2px 0; }
.km-day-temp { font-size:0.82rem; font-weight:700; color:var(--km-heading); margin:0; }
.km-day-rain { font-size:0.72rem; color:var(--km-text-muted); margin:2px 0 0; }

/* ---- crop doctor: upload card ---- */
.km-upload-hint { color:var(--km-text-muted); font-size:0.82rem; margin:4px 0 0; }
.km-kvk-card { background: var(--km-card-bg); border:1px solid var(--km-card-border); border-radius:14px;
  padding:12px 14px; font-size:0.84rem; color:var(--km-text-muted); margin-top:12px; }

/* ---- crop doctor: dark diagnosis result card (always dark, both themes) ---- */
.km-diag-card {
  background:linear-gradient(160deg,#152018,#0d140f); border:1px solid #233028; border-radius:18px;
  padding:24px 26px; color:#eef3ec; box-shadow:0 10px 30px rgba(0,0,0,0.35); height:100%;
}
.km-diag-label { font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em;
  color:#9fb89a; margin:0 0 6px; }
.km-diag-crop { font-size:0.85rem; color:#9fb89a; margin:0 0 4px; }
.km-diag-disease { font-size:1.7rem; font-weight:800; color:#ffffff; margin:0 0 12px; line-height:1.15; }
.km-diag-badge { display:inline-block; background:rgba(34,179,90,0.18); border:1px solid rgba(34,179,90,0.5);
  color:#5ee08a; font-size:0.82rem; font-weight:700; padding:4px 12px; border-radius:999px; margin-bottom:18px; }
.km-diag-subhead { font-size:0.85rem; font-weight:700; color:#ffffff; margin:14px 0 8px; }
.km-diag-card .km-bullet { color:#dbe7d8; }
.km-diag-empty { color:#9fb89a; font-size:0.9rem; }

.km-footer { text-align:center; color:var(--km-text-muted); font-size:0.8rem; margin-top:22px; }
</style>
"""

st.markdown(BASE_CSS.replace("__VARS__", DARK_VARS if THEME == "dark" else LIGHT_VARS),
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top bar: logo | nav pills | theme + language toggles
# ---------------------------------------------------------------------------
top_left, top_center, top_right = st.columns([2.2, 3.0, 2.6], vertical_alignment="center")

with top_left:
    st.markdown(
        "<div class='km-brand-row'><div class='km-logo-badge'>🌱</div>"
        "<div><div class='km-brand-name'>KisanMitra</div>"
        "<div class='km-brand-sub'>AI Farming Assistant</div></div></div>",
        unsafe_allow_html=True,
    )

with top_center:
    page = st.segmented_control(
        "Navigate", options=["🌱 Sell Advice", "📷 Crop Doctor"],
        default="🌱 Sell Advice", required=True,
        label_visibility="collapsed", key="nav_choice",
    )

with top_right:
    r1, r2 = st.columns([1, 1.6], vertical_alignment="center")
    with r1:
        theme_label = "☀️ Light" if THEME == "light" else "🌙 Dark"
        if st.button(theme_label, key="theme_btn", help="Toggle light / dark theme",
                     use_container_width=True):
            st.session_state["theme"] = "dark" if THEME == "light" else "light"
            st.rerun()
    with r2:
        st.segmented_control(
            "Language", options=["English", "हिंदी"], default="English", required=True,
            label_visibility="collapsed", key="out_lang",
        )

if not GEMINI_API_KEY:
    st.error("No Gemini API key found. Add GEMINI_API_KEY to Streamlit secrets.")
    st.stop()

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ----------------------------- SELL ADVICE PAGE -----------------------------
if page == "🌱 Sell Advice":
    left_col, right_col = st.columns([2, 1.2], vertical_alignment="top")

    with left_col:
        i1, i2, i3 = st.columns(3)
        with i1:
            crop = st.selectbox("🌾 Crop", CROPS, index=0)
        with i2:
            state = st.selectbox("📍 State", STATES, index=STATES.index("Chhattisgarh"))
        with i3:
            district = st.text_input("🏘️ District", value="Raipur")
        st.session_state["sel_state"] = state

        ask_clicked = st.button("Ask KisanMitra", type="primary", use_container_width=True)

    question = f"I grow {crop.lower()} in {district}, {state}. Should I sell now or wait?"

    if ask_clicked:
        with st.spinner("Checking weather and mandi prices…"):
            try:
                answer = ask_agent(question)
                decision = classify_decision(answer, question)
                wx_text, wx = weather_summary(district, state)
                pr_text, pr = price_summary(crop, state, district)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.session_state["last_result"] = {
            "answer": answer, "decision": decision, "wx": wx, "pr": pr,
            "district": district, "state": state, "crop": crop,
        }

    result = st.session_state.get("last_result")

    with right_col:
        if result:
            decision = result["decision"]
            headline = _pick(decision["headline_en"], decision["headline_hi"])
            summary = _pick(decision["summary_en"], decision["summary_hi"])
            pct = decision.get("confidence_pct", 70)
            accent = "#16a34a" if decision["verdict"] in ("SELL", "WAIT") else "#64748b"
            st.markdown(
                f"<div class='km-rec-card' style='border-left:5px solid {accent}'>"
                f"<p class='km-rec-label'>Recommendation</p>"
                f"<p class='km-rec-headline' style='color:{accent}'>{html.escape(headline)}</p>"
                f"<p class='km-rec-summary'>{html.escape(summary)}</p>"
                f"<div class='km-conf-row'><span>Confidence</span><span>{pct}%</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(pct / 100)
        else:
            st.markdown(
                "<div class='km-rec-card' style='border-left:5px solid var(--km-card-border)'>"
                "<p class='km-rec-label'>Recommendation</p>"
                "<p class='km-small'>Ask KisanMitra to see your sell / wait recommendation here.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    if result:
        decision = result["decision"]
        wx, pr = result["wx"], result["pr"]
        c1, c2, c3 = st.columns(3)

        with c1:
            if wx:
                tomo = wx["days"][1] if len(wx["days"]) > 1 else wx["days"][0]
                st.markdown(
                    f"<div class='km-card'><p class='km-card-head'>🌧️ Weather Outlook</p>"
                    f"<p class='km-big'>{wx['rain3']} mm</p>"
                    f"<p class='km-small'>Rain expected next 3 days</p>"
                    f"<p class='km-small'>Tomorrow {tomo['tmin']}–{tomo['tmax']}°C • "
                    f"{tomo['rain_pct']}% chance</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='km-card'><p class='km-card-head'>🌧️ Weather Outlook</p>"
                            "<p class='km-small'>Unavailable right now.</p></div>",
                            unsafe_allow_html=True)

        with c2:
            if pr["rows"]:
                top = pr["rows"][0]
                st.markdown(
                    f"<div class='km-card'><p class='km-card-head'>₹ Market Price</p>"
                    f"<p class='km-big'>₹{top['modal']} <span style='font-size:0.95rem;font-weight:500'>/qtl</span></p>"
                    f"<p class='km-small'>{html.escape(top['market'])}</p>"
                    f"<p class='km-small'>{html.escape(top['date'])} • range ₹{top['lo']}–{top['hi']}</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"<div class='km-card'><p class='km-card-head'>₹ Market Price</p>"
                            f"<p class='km-small'>{html.escape(pr['summary'])}</p></div>",
                            unsafe_allow_html=True)

        with c3:
            bullets = "".join(
                f"<div class='km-bullet'><span class='km-bullet-icon'>✅</span><span>{html.escape(b)}</span></div>"
                for b in _pick(decision["reasoning_en"], decision["reasoning_hi"])
            ) or "<p class='km-small'>No extra reasoning returned.</p>"
            st.markdown(
                f"<div class='km-card'><p class='km-card-head'>💡 Why this advice?</p>{bullets}</div>",
                unsafe_allow_html=True,
            )

        with st.expander(f"📊 Live Market & Weather Data · {result['district']}, {result['state']}"):
            dcol, wcol = st.columns([1.1, 1.4])
            with dcol:
                st.markdown("<p class='km-card-head' style='margin-bottom:6px'>Recent mandi records</p>",
                            unsafe_allow_html=True)
                if pr["rows"]:
                    for r in pr["rows"]:
                        st.markdown(
                            f"<div class='km-row'><span>{html.escape(r['market'])} · {html.escape(r['date'])}</span>"
                            f"<span><b>₹{r['modal']}</b> ({r['lo']}–{r['hi']})</span></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("<p class='km-small'>No recent records.</p>", unsafe_allow_html=True)
            with wcol:
                st.markdown("<p class='km-card-head' style='margin-bottom:6px'>5-day forecast</p>",
                            unsafe_allow_html=True)
                if wx:
                    daycols = st.columns(len(wx["days"]))
                    for col, x in zip(daycols, wx["days"]):
                        icon = "🌧️" if x["rain"] >= 5 else ("🌦️" if x["rain"] > 0 else "☀️")
                        with col:
                            st.markdown(
                                f"<div class='km-day'><p class='km-day-name'>{html.escape(x['date'][5:])}</p>"
                                f"<p class='km-day-icon'>{icon}</p>"
                                f"<p class='km-day-temp'>{x['tmin']}–{x['tmax']}°C</p>"
                                f"<p class='km-day-rain'>{x['rain']}mm · {x['rain_pct']}%</p></div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.markdown("<p class='km-small'>No forecast available.</p>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<p class='km-small' style='text-align:center;margin-top:8px'>"
            "Pick a crop, state and district, then tap <b>Ask KisanMitra</b> to see your "
            "weather outlook, market price, and reasoning.</p>",
            unsafe_allow_html=True,
        )

# ----------------------------- CROP DOCTOR PAGE -----------------------------
else:
    dl_col, dr_col = st.columns([1, 1.1], vertical_alignment="top")

    with dl_col:
        st.markdown("<div class='km-card-head' style='margin-bottom:2px'>📤 Upload Crop Photo</div>"
                    "<p class='km-upload-hint'>Drag &amp; drop or click to upload — "
                    "JPG, PNG, WEBP up to 10MB.</p>", unsafe_allow_html=True)
        photo = st.file_uploader("Crop photo", type=["jpg", "jpeg", "png", "webp"],
                                  max_upload_size=10, label_visibility="collapsed")
        if photo is not None:
            st.image(photo, caption="Your photo", use_container_width=True)
            diagnose_clicked = st.button("Diagnose", type="primary", use_container_width=True)
        else:
            diagnose_clicked = False

        st.markdown(
            "<div class='km-kvk-card'>⚠️ AI guidance, not a substitute for your local "
            "Krishi Vigyan Kendra. Confirm before spraying chemicals.</div>",
            unsafe_allow_html=True,
        )

    if diagnose_clicked:
        with st.spinner("Looking at your crop…"):
            try:
                st.session_state["last_diagnosis"] = diagnose_image(photo.getvalue(), photo.type or "image/jpeg")
            except Exception as e:
                st.session_state["last_diagnosis"] = None
                st.error(f"Could not diagnose: {e}")

    diag = st.session_state.get("last_diagnosis")

    with dr_col:
        if diag:
            crop_line = (f"<p class='km-diag-crop'>Crop: {html.escape(diag['crop'])}</p>"
                         if diag.get("crop") and diag["crop"] != "Unclear" else "")
            todo = "".join(
                f"<div class='km-bullet'><span class='km-bullet-icon'>✅</span><span>{html.escape(s)}</span></div>"
                for s in diag["what_to_do"]
            ) or "<p class='km-small' style='color:#9fb89a'>No steps returned.</p>"
            precautions = "".join(
                f"<div class='km-bullet'><span class='km-bullet-icon'>⚠️</span><span>{html.escape(s)}</span></div>"
                for s in diag["precautions"]
            ) or "<p class='km-small' style='color:#9fb89a'>None noted.</p>"
            st.markdown(
                f"<div class='km-diag-card'>"
                f"<p class='km-diag-label'>Detected Issue</p>"
                f"{crop_line}"
                f"<p class='km-diag-disease'>{html.escape(diag['disease'])}</p>"
                f"<span class='km-diag-badge'>Confidence: {diag['confidence_pct']}%</span>"
                f"<p class='km-diag-subhead'>✅ What To Do</p>{todo}"
                f"<p class='km-diag-subhead'>⚠️ Precautions</p>{precautions}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='km-diag-card'><p class='km-diag-label'>Diagnosis Result</p>"
                "<p class='km-diag-empty'>Upload a clear daylight photo of the affected leaf "
                "and tap Diagnose to see the result here.</p></div>",
                unsafe_allow_html=True,
            )

st.markdown(
    "<p class='km-footer'>Data: Open-Meteo (weather) + data.gov.in AGMARKNET (mandi prices). "
    "Prices may be from nearby markets when local data is unavailable.</p>",
    unsafe_allow_html=True,
)
