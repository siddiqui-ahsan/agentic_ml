import os
import json
import re
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed   # ← neu
import pandas as pd
from tqdm import tqdm

# Disable tqdm on Railway (no terminal → would spam logs)
_DISABLE_PROGRESS = not os.isatty(1)

# ── 1. Define your flags ──────────────────────────────────────────────────────

LUXURY_FLAGS = {
    "has_luxury_finishes":  "mentions marble, hardwood, high-end appliances, designer furniture",
    "has_waterfront_view":  "mentions water view, river, ocean, lake, harbor, skyline",
    "has_outdoor_space":    "mentions terrace, balcony, rooftop, garden, patio",
    "needs_renovation":     "mentions repairs needed, dated, worn, basic, simple",
    "is_entire_home":       "entire apartment or house, not shared",
    "has_doorman":          "mentions doorman, concierge, staffed building",
    "has_premium_location": "mentions central park, tribeca, manhattan, soho, upper east",
    "is_newly_renovated":   "mentions newly renovated, brand new, modern, just updated",
    "has_amenities":        "mentions gym, pool, spa, rooftop lounge",
    "is_budget_language":   "mentions cozy, affordable, simple, basic, budget, functional",
}

# ── 2. The prompt — strict JSON only ─────────────────────────────────────────

def build_prompt(description: str) -> str:
    flags_spec = "\n".join(
        f'  "{k}": true/false  // {v}'
        for k, v in LUXURY_FLAGS.items()
    )
    return f"""You are a real estate data extraction API. 
Analyze this Airbnb listing description and return ONLY a JSON object with these boolean fields.
No explanation, no markdown, no preamble — raw JSON only.

Fields:
{flags_spec}

Description:
\"\"\"{description[:800]}\"\"\"

JSON:"""

# ── 3. Call Ollama ────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = "llama3.1:8b") -> Optional[str]:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 200,
                }
            },
            timeout=30
        )
        return response.json().get("response", "")
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

# ── 4. Parse JSON robustly ────────────────────────────────────────────────────

def parse_flags(raw: Optional[str]) -> dict:
    if not raw:
        return _default_flags()
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return _default_flags()

def _default_flags() -> dict:
    return {k: False for k in LUXURY_FLAGS}

# ── 5. Process a single row (used by thread pool) ────────────────────────────

def _process_row(args: tuple) -> tuple[int, dict]:
    """
    Verarbeitet eine einzelne Zeile.
    Gibt (original_index, flags_dict) zurück.
    Wird von ThreadPoolExecutor parallel aufgerufen.
    """
    i, row, text_col, model = args
    description = str(row.get(text_col, "") or "")

    if len(description.strip()) < 20:
        return i, _default_flags()

    prompt = build_prompt(description)
    raw    = call_ollama(prompt, model=model)
    flags  = parse_flags(raw)

    for key in LUXURY_FLAGS:
        flags.setdefault(key, False)

    return i, flags

# ── 6. Extract for a whole DataFrame — parallel ───────────────────────────────

def extract_flags_batch(
    df: pd.DataFrame,
    text_col: str = "description",
    model: str = "llama3.1:8b",
    max_workers: int = 6,           # ← 6 parallele Threads; bei Ollama-Überlastung auf 4 senken
) -> pd.DataFrame:
    """
    Schickt max_workers Zeilen gleichzeitig an Ollama.
    Ollama verarbeitet sie intern sequenziell, aber die Netzwerk-/Wartezeit
    wird überlappt → ca. 4-6× schneller als sequenziell.

    Reihenfolge bleibt erhalten (results[i] = flags für Zeile i).
    """

    rows = list(df.iterrows())          # [(pandas_index, row_series), ...]
    results = [None] * len(rows)        # Platzhalter — Reihenfolge wird bewahrt

    # Argumente für jeden Thread vorbereiten
    tasks = [
        (i, row, text_col, model)
        for i, (_, row) in enumerate(rows)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Alle Tasks einreichen
        future_to_idx = {
            executor.submit(_process_row, task): task[0]
            for task in tasks
        }

        # Ergebnisse einsammeln sobald sie fertig sind (as_completed = schnellste zuerst)
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(rows),
            desc="LLM extraction",
            disable=_DISABLE_PROGRESS,
        ):
            i, flags = future.result()
            results[i] = flags                  # an die richtige Position schreiben

            # Railway-freundliches Logging alle 100 Zeilen
            if _DISABLE_PROGRESS and (i + 1) % 100 == 0:
                print(f"  LLM extraction: {i + 1}/{len(rows)} rows done")

    flags_df = pd.DataFrame(results, index=df.index)
    return flags_df.astype(int)
