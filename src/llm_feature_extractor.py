 
import os
import json
import re
import requests
from typing import Optional
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
        # Single line — no tqdm spam
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
 
# ── 5. Extract for a whole DataFrame ─────────────────────────────────────────
 
def extract_flags_batch(
    df: pd.DataFrame,
    text_col: str = "description",
    model: str = "llama3.1:8b",
) -> pd.DataFrame:
 
    results = []
    total = len(df)
 
    for i, (_, row) in enumerate(tqdm(
        df.iterrows(),
        total=total,
        desc="LLM extraction",
        disable=_DISABLE_PROGRESS,   # silent on Railway
    )):
        description = str(row.get(text_col, "") or "")
 
        if len(description.strip()) < 20:
            results.append(_default_flags())
            continue
 
        prompt = build_prompt(description)
        raw    = call_ollama(prompt, model=model)
        flags  = parse_flags(raw)
 
        for key in LUXURY_FLAGS:
            flags.setdefault(key, False)
 
        results.append(flags)
 
        # Single progress log every 100 rows — Railway-friendly
        if not _DISABLE_PROGRESS and (i + 1) % 100 == 0:
            print(f"  LLM extraction: {i + 1}/{total} rows done")
 
    flags_df = pd.DataFrame(results, index=df.index)
    return flags_df.astype(int)