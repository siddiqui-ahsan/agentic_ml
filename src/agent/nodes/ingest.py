# agent/nodes/ingest.py

import pandas as pd
from src.agent.state import AgentState

def ingest_node(state: AgentState) -> AgentState:
    """Load CSVs — tries multiple encodings, survives broken files."""
    print("▶ Node: ingest")
    
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            train_df = pd.read_csv(state["train_path"], encoding=encoding)
            test_df  = pd.read_csv(state["test_path"],  encoding=encoding)
            
            return {**state,
                "train_df":    train_df,
                "test_df":     test_df,
                "current_node": "ingest",
                "errors":      state["errors"],
            }
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return {**state,
                "errors": state["errors"] + [f"Ingest failed: {e}"],
            }
    
    return {**state, "errors": state["errors"] + ["All encodings failed"]}