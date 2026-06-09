# src/agent/nodes/predict.py

import pandas as pd
from src.agent.nodes.ml_train import normalize_columns, safe_select_features, ALL_FEATURE_COLS


def predict_node(state: dict) -> dict:
    print("▶ Node: predict")

    pipeline  = state["pipeline"]
    flag_cols = state.get("flag_cols", [])

    # Step 1: Normalize test_df columns (same as training)
    test_df = normalize_columns(state["test_df"])

    # Step 2: Merge LLM flags for test set if available
    test_flags = state.get("test_flags")
    if not state["llm_failed"] and test_flags is not None:
        test_df = pd.concat([test_df, test_flags], axis=1)

    # Step 3: Safely select features — same columns as training, no KeyError
    feature_cols = ALL_FEATURE_COLS + flag_cols
    X_test = safe_select_features(test_df, feature_cols)

    print(f"  Predicting on {len(X_test)} rows")

    # Step 4: Predict
    predictions = pipeline.predict(X_test)

    # Step 5: Keep property_id for output — inject as 0..n if missing
    if "property_id" in test_df.columns:
        property_ids = test_df["property_id"]
    else:
        print("  [predict] 'property_id' missing — using row index as fallback")
        property_ids = pd.Series(range(len(test_df)), name="property_id")

    return {**state,
        "predictions":  predictions,
        "property_ids": property_ids,
        "current_node": "predict",
    }
