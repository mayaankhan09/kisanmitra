"""
KisanMitra — mandi data layer (Phase 1 -> feeds Phase 2)
========================================================
Why this exists:
  The live data.gov.in API works, but any single pull is dominated by whichever
  states reported that hour. Chhattisgarh is sparsely reported, so one snapshot
  may have ~1 CG row. The fix: keep appending daily snapshots into one local
  store, de-duplicated. Coverage grows over days. The agent then reads this
  store via get_mandi_price(), which falls back gracefully so it ALWAYS returns
  something useful for a demo.

Usage:
  1. Each time you download a fresh snapshot CSV from data.gov.in (browser),
     drop it into the ./snapshots/ folder (any filename).
  2. Run rebuild_store() once to merge them all into mandi_store.csv.
  3. Your agent calls get_mandi_price(...).
"""

import os
import glob
import pandas as pd

SNAP_DIR = "snapshots"
STORE = "mandi_store.csv"

# Districts near Chhattisgarh (for regional fallback when CG itself is thin).
# Data spells the state "Chattisgarh" (one 'h') — handled in normalisation.
NEIGHBOUR_STATES = ["Chattisgarh", "Madhya Pradesh", "Odisha", "Maharashtra",
                    "Telangana", "Jharkhand", "Uttar Pradesh"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Fix column names and known spelling quirks."""
    df = df.copy()
    df.columns = [c.strip().lower().replace("_x0020_", "_").replace(" ", "_")
                  for c in df.columns]
    # Standardise the Chhattisgarh spelling so callers can use the normal spelling
    if "state" in df.columns:
        df["state"] = df["state"].replace(
            {"Chattisgarh": "Chhattisgarh", "Chhatisgarh": "Chhattisgarh"})
    # Coerce prices to numbers
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rebuild_store() -> pd.DataFrame:
    """Merge every CSV in ./snapshots/ into one de-duplicated store file."""
    files = glob.glob(os.path.join(SNAP_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs in ./{SNAP_DIR}/. Put your downloaded snapshot(s) there.")
    frames = [_normalise(pd.read_csv(f)) for f in files]
    store = pd.concat(frames, ignore_index=True)
    # De-dupe on the natural key of a price record
    key = ["state", "district", "market", "commodity", "variety", "arrival_date"]
    key = [k for k in key if k in store.columns]
    before = len(store)
    store = store.drop_duplicates(subset=key)
    store.to_csv(STORE, index=False)
    print(f"Merged {len(files)} file(s): {before} -> {len(store)} unique rows.")
    print(f"States covered: {store['state'].nunique()}  |  saved to {STORE}")
    return store


def load_store() -> pd.DataFrame:
    """Load the merged store; if it doesn't exist yet, build it."""
    if not os.path.exists(STORE):
        return rebuild_store()
    return _normalise(pd.read_csv(STORE))


def _rows_to_records(df: pd.DataFrame, limit: int):
    df = df.sort_values("arrival_date", ascending=False).head(limit)
    return df[["market", "district", "state", "commodity", "variety",
               "arrival_date", "min_price", "max_price", "modal_price"]].to_dict("records")


def get_mandi_price(commodity: str,
                    state: str = "Chhattisgarh",
                    district: str | None = None,
                    limit: int = 15,
                    df: pd.DataFrame | None = None) -> dict:
    """Return mandi prices for a commodity with graceful geographic fallback.
    Tries: district -> state -> neighbouring region -> all-India.
    Always returns the most local data available, and says which level it used.
    Prices are Rs/quintal (100 kg)."""
    if df is None:
        df = load_store()

    base = df[df["commodity"].str.contains(commodity, case=False, na=False)]
    if base.empty:
        return {"commodity": commodity, "scope": "none", "records": [],
                "summary": f"No '{commodity}' anywhere in the local store yet. "
                           f"Try a fresh snapshot or a different crop."}

    # 1) district level
    if district:
        d = base[base["district"].str.contains(district, case=False, na=False)]
        if not d.empty:
            return _pack(commodity, f"{district} district", d, limit)
    # 2) state level
    s = base[base["state"].str.contains(state, case=False, na=False)]
    if not s.empty:
        return _pack(commodity, state, s, limit)
    # 3) neighbouring region
    region = base[base["state"].isin([_normalise_state(x) for x in NEIGHBOUR_STATES])]
    if not region.empty:
        return _pack(commodity, "nearby states", region, limit,
                     note=f"No {commodity} in {state} yet; showing nearby-state prices as a proxy.")
    # 4) all-India
    return _pack(commodity, "all-India", base, limit,
                 note=f"No {commodity} near {state} yet; showing national prices as a reference.")


def _normalise_state(name: str) -> str:
    return {"Chattisgarh": "Chhattisgarh"}.get(name, name)


def _pack(commodity, scope, df, limit, note=""):
    records = _rows_to_records(df, limit)
    modal = [r["modal_price"] for r in records if pd.notna(r["modal_price"])]
    if modal:
        lo, hi = int(min(modal)), int(max(modal))
        latest = records[0]
        summary = (f"{commodity} ({scope}): {len(records)} records. "
                   f"Modal Rs {lo}-{hi}/quintal. "
                   f"Latest: {latest['market']} on {latest['arrival_date']} "
                   f"at Rs {int(latest['modal_price'])}.")
    else:
        summary = f"{commodity} ({scope}): {len(records)} records, prices missing."
    if note:
        summary = note + " " + summary
    return {"commodity": commodity, "scope": scope, "records": records,
            "summary": summary}


if __name__ == "__main__":
    store = rebuild_store()
    print("\n--- TEST: Paddy in Chhattisgarh ---")
    print(get_mandi_price("Paddy", state="Chhattisgarh", district="Raipur", df=store)["summary"])
    print("\n--- TEST: Onion (well-covered nationally) ---")
    print(get_mandi_price("Onion", state="Chhattisgarh", df=store)["summary"])
    print("\n--- TEST: Tomato ---")
    print(get_mandi_price("Tomato", state="Chhattisgarh", df=store)["summary"])
