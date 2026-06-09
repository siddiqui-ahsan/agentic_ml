# src/agent/nodes/output.py

import os
import pandas as pd


def output_node(state: dict) -> dict:
    print("▶ Node: output")

    predictions  = state["predictions"]
    property_ids = state["property_ids"]
    output_path  = state.get("output_path", "data/predictions.csv")

    # Build output DataFrame
    output_df = pd.DataFrame({
        "property_id":           property_ids.values,
        "predicted_price_tier":  predictions,
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_df.to_csv(output_path, index=False)

    print(f"  Saved {len(output_df)} predictions → {output_path}")
    print(f"  Distribution:\n{output_df['predicted_price_tier'].value_counts().to_string()}")

    if state["errors"]:
        print(f"  Warnings during run: {state['errors']}")

    return {**state,
        "output_path":  output_path,
        "current_node": "output",
    }
