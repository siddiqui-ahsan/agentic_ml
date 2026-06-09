import os
import json
import re
import time
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv   # liest .env Datei automatisch ein
 
load_dotenv()  # laedt OPENROUTER_API_KEY aus .env
 
# Disable tqdm on Railway (no terminal -> would spam logs)
_DISABLE_PROGRESS = not os.isatty(1)
 
# -- 1. Define your flags -----------------------------------------------------
 
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
 
# -- 2. The prompt - strict JSON only -----------------------------------------
 
def build_prompt(description: str) -> str:
    flags_spec = "\n".join(
        f'  "{k}": true/false  // {v}'
        for k, v in LUXURY_FLAGS.items()
    )
    return f"""You are a real estate data extraction API. 
Analyze this Airbnb listing description and return ONLY a JSON object with these boolean fields.
No explanation, no markdown, no preamble - raw JSON only.
 
Fields:
{flags_spec}
 
Description:
\"\"\"{description[:800]}\"\"\"
 
JSON:"""
 
# -- 3. Call OpenRouter -------------------------------------------------------
 
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
 
def call_openrouter(prompt: str) -> Optional[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY nicht gesetzt!\n"
            "Entweder: export OPENROUTER_API_KEY='sk-or-...'\n"
            "Oder: .env Datei mit OPENROUTER_API_KEY=sk-or-... erstellen"
        )
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model":       OPENROUTER_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens":  200,
            },
            timeout=30,
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenRouter error: {e}")
        return None
 
# -- 4. Parse JSON robustly ---------------------------------------------------
 
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
 
# -- 5. Process a single row (used by thread pool) ----------------------------
 
def _process_row(args: tuple) -> tuple[int, dict]:
    i, row, text_col = args
    description = str(row.get(text_col, "") or "")
 
    if len(description.strip()) < 20:
        return i, _default_flags()
 
    prompt = build_prompt(description)
 
    # Rate limit schutz: free tier erlaubt ~20 req/min
    # Mit max_workers=5 und 0.3s sleep bleiben wir sicher darunter
    time.sleep(0.3)
 
    raw   = call_openrouter(prompt)
    flags = parse_flags(raw)
 
    for key in LUXURY_FLAGS:
        flags.setdefault(key, False)
 
    return i, flags
 
# -- 6. Extract for a whole DataFrame - parallel ------------------------------
 
def extract_flags_batch(
    df: pd.DataFrame,
    text_col: str = "description",
    max_workers: int = 5,    # free tier: 20 req/min -> 5 threads x 0.3s sleep = ~17 req/min
) -> pd.DataFrame:
    """
    Schickt max_workers Zeilen gleichzeitig an OpenRouter.
    Rate limit des free tiers (20 req/min) wird durch time.sleep(0.3) eingehalten.
    Reihenfolge bleibt erhalten (results[i] = flags fuer Zeile i).
    Erwartete Zeit: 1800 Zeilen / 17 req/min = ~17 Minuten
    """
 
    rows    = list(df.iterrows())
    results = [None] * len(rows)
 
    tasks = [
        (i, row, text_col)
        for i, (_, row) in enumerate(rows)
    ]
 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_row, task): task[0]
            for task in tasks
        }
 
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(rows),
            desc="LLM extraction (OpenRouter)",
            disable=_DISABLE_PROGRESS,
        ):
            i, flags = future.result()
            results[i] = flags
 
            if _DISABLE_PROGRESS and (i + 1) % 100 == 0:
                print(f"  LLM extraction: {i + 1}/{len(rows)} rows done")
 
    flags_df = pd.DataFrame(results, index=df.index)
    return flags_df.astype(int)