# llm_feature_extractor.py
import json
import re
import requests
from typing import Optional
import pandas as pd
from tqdm import tqdm

# ── 1. Define your flags ──────────────────────────────────────────────────────

LUXURY_FLAGS = {
    "has_luxury_finishes":   "mentions marble, hardwood, high-end appliances, designer furniture",
    "has_waterfront_view":   "mentions water view, river, ocean, lake, harbor, skyline",
    "has_outdoor_space":     "mentions terrace, balcony, rooftop, garden, patio",
    "needs_renovation":      "mentions repairs needed, dated, worn, basic, simple",
    "is_entire_home":        "entire apartment or house, not shared",
    "has_doorman":           "mentions doorman, concierge, staffed building",
    "has_premium_location":  "mentions central park, tribeca, manhattan, soho, upper east",
    "is_newly_renovated":    "mentions newly renovated, brand new, modern, just updated",
    "has_amenities":         "mentions gym, pool, spa, rooftop lounge",
    "is_budget_language":    "mentions cozy, affordable, simple, basic, budget, functional",
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

def call_ollama(prompt: str, model: str = "llama3") -> Optional[str]:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,   # deterministic — no creativity needed
                    "num_predict": 200,   # flags fit in ~150 tokens
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
    """Tries to extract JSON even if the LLM added extra text around it."""
    if not raw:
        return _default_flags()
    
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block within the response
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Full fallback — return neutral defaults
    return _default_flags()

def _default_flags() -> dict:
    """Safe fallback: all flags False — better than crashing."""
    return {k: False for k in LUXURY_FLAGS}

# ── 5. Extract for a whole DataFrame ─────────────────────────────────────────

def extract_flags_batch(
    df: pd.DataFrame,
    text_col: str = "description",
    model: str = "llama3",
    batch_size: int = 50,        # pause every 50 rows to avoid OOM
) -> pd.DataFrame:
    
    results = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="LLM extraction"):
        description = str(row.get(text_col, "") or "")
        
        if len(description.strip()) < 20:
            # Too short to be useful — use defaults, don't waste tokens
            results.append(_default_flags())
            continue
        
        prompt = build_prompt(description)
        raw = call_ollama(prompt, model=model)
        flags = parse_flags(raw)
        
        # Ensure all expected keys exist (LLM might skip some)
        for key in LUXURY_FLAGS:
            flags.setdefault(key, False)
        
        results.append(flags)
    
    flags_df = pd.DataFrame(results, index=df.index)
    
    # Cast all to int (0/1) for sklearn compatibility
    return flags_df.astype(int)