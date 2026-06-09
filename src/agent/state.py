# src/agent/state.py

import pandas as pd
from typing import TypedDict, Optional


class AgentState(TypedDict):
    # Data
    train_df:        Optional[pd.DataFrame]
    test_df:         Optional[pd.DataFrame]
    train_flags:     Optional[pd.DataFrame]
    test_flags:      Optional[pd.DataFrame]
    predictions:     Optional[pd.Series]

    # Control flow
    errors:          list[str]
    retry_count:     int
    llm_failed:      bool
    current_node:    str

    # Config
    train_path:      str
    test_path:       str
    output_path:     str
