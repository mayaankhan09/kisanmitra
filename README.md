# 🌾 KisanMitra — किसान मित्र

**An AI agent that helps small farmers in Chhattisgarh decide when to sell, harvest, or treat their crop.**

KisanMitra answers a farmer's plain-language question — in Hindi or English — by reasoning
across **live weather** and **live mandi (market) prices**, and can also **diagnose crop
disease from a photo**. It turns public data that already exists, but is scattered and hard
to use, into a single clear recommendation a farmer can act on.

> Built for the Kaggle × Google *5-Day AI Agents Intensive* capstone — **Agents for Good** track.

🔗 **Live app:** [your Streamlit URL here]
🎥 **Demo video:** [your YouTube URL here]

---

## Why this exists

A small farmer growing paddy near Raipur faces two questions every season: *is the price
good enough to sell now?* and *will the weather let me harvest in time?* The data to answer
both is published daily by the government and weather services — but it's spread across
portals, in English, in formats a farmer with a basic phone can't easily combine. So
decisions get made on guesswork. KisanMitra closes that gap.

---

## What it does

**💬 Sell / harvest advisor (the agent).**
The farmer asks something like *"मेरे पास रायपुर में धान है, अभी बेचूं या रुकूं?"*. The agent
autonomously calls a weather tool and a mandi-price tool, reasons across both, and replies
with one clear recommendation — citing the real numbers it used, in the farmer's language.

**📷 Crop doctor (multimodal).**
The farmer uploads a photo of an affected leaf. Using Gemini's vision capability, KisanMitra
identifies the likely disease, states its confidence honestly, and suggests low-cost local
remedies — always advising confirmation with the local Krishi Vigyan Kendra before spraying.

---

## How it works

```
Farmer's question
      │
      ▼
  Gemini agent  ──►  get_weather_for_district()   (Open-Meteo API)
 (function-calling)   get_crop_price()             (data.gov.in AGMARKNET)
      │
      ▼
 One recommendation, citing real weather + price data
```

The core is a **Gemini function-calling agent**. Two Python functions are passed to the model
as tools; the model decides when to call them, runs them, and synthesises the results. The
system prompt enforces the behaviour that makes it a true agent: it must weigh **both** weather
and price before deciding, reply in the farmer's language, and cite its numbers.

The crop doctor is a separate **multimodal** Gemini call, because diagnosing a leaf is a vision
task rather than a data lookup.

### Engineering for unreliable real-world data

The data.gov.in feed is real but slow and unevenly covered — any single snapshot is dominated
by whichever states reported that day, so Chhattisgarh is often sparse. The data also spells
the state "Chattisgarh", stores prices as strings, and contains occasional junk rows. The
`mandi_data.py` layer handles all of this: it accumulates daily snapshots into one
de-duplicated store, normalises the quirks, filters invalid prices, and **falls back
gracefully** (district → state → neighbouring states → all-India) so the agent always returns
something useful and honest — and the live demo never hangs on a government server.

---

## Tech stack

- **Model:** Google Gemini 2.5 Flash (`google-genai` SDK)
- **Agent pattern:** Gemini automatic function-calling
- **Weather:** Open-Meteo API (free, no key)
- **Prices:** data.gov.in AGMARKNET daily mandi feed
- **UI / hosting:** Streamlit + Streamlit Community Cloud
- **Languages:** Hindi & English

---

## Project structure

```
.
├── streamlit_app.py        # The web app (advisor + crop doctor tabs)
├── agent_core.py           # Notebook version of the agent (function-calling core)
├── mandi_data.py           # Data layer: load/merge snapshots, normalise, fallback
├── phase1_data_check.py    # Standalone script that validates both data sources
├── snapshots/              # Daily mandi-price CSV snapshots (de-duplicated on load)
│   └── snapshot_2026-06-21.csv
├── requirements.txt
├── secrets.toml.example    # Shows the secrets format (real keys go in .streamlit/)
└── README.md
```

---

## Run it locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add your Gemini API key**
Get a free key from [Google AI Studio](https://aistudio.google.com). Create a file at
`.streamlit/secrets.toml` containing:
```toml
GEMINI_API_KEY = "your-key-here"
```
(This file is gitignored and never committed.)

**3. Run**
```bash
streamlit run streamlit_app.py
```

The app opens in your browser. Pick a crop and district and ask whether to sell or wait, then
try the crop doctor tab with a leaf photo.

---

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app from the repo with
   `streamlit_app.py` as the main file.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Deploy — you'll get a public URL.

---

## Keeping data fresh

To grow mandi coverage, download a new snapshot CSV from the data.gov.in AGMARKNET resource
and drop it into `snapshots/`. The data layer merges and de-duplicates all snapshots
automatically on load, so coverage improves over time.

---

## Limitations & next steps

- **Mandi coverage** depends on what AGMARKNET reports; rural markets are under-covered.
- **Disease diagnosis** is AI guidance, not a lab diagnosis — always confirm before spraying.
- **Languages:** Hindi + English today; Chhattisgarhi and voice input are natural next steps.
- **No price forecasting yet** — the agent reasons over current/recent prices, not predicted ones.

---

## Acknowledgements

- Weather data: [Open-Meteo](https://open-meteo.com)
- Mandi prices: [data.gov.in](https://data.gov.in) AGMARKNET (Ministry of Agriculture & Farmers Welfare)
- Built with [Google Gemini](https://ai.google.dev) and [Streamlit](https://streamlit.io)

---

*KisanMitra — putting the information farmers already have a right to, into a form they can actually use.*
