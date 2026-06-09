# agent/nodes/schema.py

import pandas as pd
import numpy as np
from src.agent.graph import AgentState

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

def schema_repair_node(state: AgentState) -> AgentState:
    print("▶ Node: schema_repair")

    train_df = _repair(state["train_df"])
    test_df  = _repair(state["test_df"])

    return {**state,
        "train_df":     train_df,
        "test_df":      test_df,
        "current_node": "schema",
    }

def _repair(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Normalize column names
    df = df.copy()
    df.columns = (df.columns
                    .str.lower()
                    .str.strip()
                    .str.replace(r'[\s\-]+', '_', regex=True))

    # 2. Fuzzy alias matching
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: canonical})
                    print(f"  Renamed '{alias}' → '{canonical}'")
                    break

    # 3. Inject missing columns as NaN — imputer handles the rest
    all_expected = list(COLUMN_ALIASES.keys())
    for col in all_expected:
        if col not in df.columns:
            df[col] = np.nan
            print(f"  Injected missing column: '{col}' as NaN")

    return df