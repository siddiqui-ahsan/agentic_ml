# src/agent/nodes/data.py
# Kombiniert: ingest.py + schema.py + llm_extract.py

import pandas as pd
import numpy as np
from src.agent.state import AgentState

# ── 1. Column Aliases (Schema Repair) ────────────────────────────────────────

COLUMN_ALIASES = {
    "description":        ["desc", "listing_description", "name", "about",
                           "summary", "neighborhood_overview", "space"],
    "neighbourhood_group": ["borough", "area", "neighborhood_group", "district"],
    "neighbourhood":      ["neighborhood", "location", "zone"],
    "room_type":          ["room", "type", "property_type", "accommodation"],
    "property_id":        ["id", "listing_id", "pid"],
    "latitude":           ["lat"],
    "longitude":          ["lon", "lng", "long"],
    "minimum_nights":     ["min_nights", "minimum_stay"],
    "number_of_reviews":  ["num_reviews", "reviews", "review_count"],
    "availability_365":   ["availability", "avail_365"],
    "calculated_host_listings_count": ["host_listings", "host_count"],
}

def _repair(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize column names
    df.columns = (df.columns
                    .str.lower()
                    .str.strip()
                    .str.replace(r'[\s\-]+', '_', regex=True))
    # Fuzzy alias matching
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: canonical})
                    print(f"  Renamed '{alias}' → '{canonical}'")
                    break
    # Inject missing columns as NaN
    for col in COLUMN_ALIASES.keys():
        if col not in df.columns:
            df[col] = np.nan
            print(f"  Injected missing column: '{col}' as NaN")
    return df

# ── 2. Ingest Node ────────────────────────────────────────────────────────────

def ingest_node(state: AgentState) -> AgentState:
    """Load CSVs — tries multiple encodings, survives broken files."""
    print("▶ Node: ingest")

    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            train_df = pd.read_csv(state["train_path"], encoding=encoding)
            test_df  = pd.read_csv(state["test_path"],  encoding=encoding)
            return {**state,
                "train_df":     train_df,
                "test_df":      test_df,
                "current_node": "ingest",
                "errors":       state["errors"],
            }
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return {**state,
                "errors": state["errors"] + [f"Ingest failed: {e}"],
            }

    return {**state, "errors": state["errors"] + ["All encodings failed"]}

# ── 3. Schema Repair Node ─────────────────────────────────────────────────────

def schema_repair_node(state: AgentState) -> AgentState:
    print("▶ Node: schema_repair")
    return {**state,
        "train_df":     _repair(state["train_df"]),
        "test_df":      _repair(state["test_df"]),
        "current_node": "schema",
    }

# ── 4. LLM Extract Node ───────────────────────────────────────────────────────
# LLM flag extraction removed — BERT + TF-IDF in ml_train handles text directly.
# This node is kept as a pass-through so graph.py does not need to change.

def llm_extract_node(state: AgentState) -> AgentState:
    print("▶ Node: llm_extract (skipped — BERT handles text in ml_train)")
    return {**state,
        "train_flags":  None,
        "test_flags":   None,
        "llm_failed":   True,   # signals ml_train to use BERT+TF-IDF branch
        "current_node": "llm_extract",
    }
