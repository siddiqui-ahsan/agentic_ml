# agent/graph.py
import os
import pandas as pd
import numpy as np
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agent.nodes.ingest      import ingest_node
from agent.nodes.schema      import schema_repair_node
from agent.nodes.llm_extract import llm_extract_node
from agent.nodes.ml_train    import ml_train_node
from agent.nodes.predict     import predict_node
from agent.nodes.output      import output_node

# ── 1. State Definition ───────────────────────────────────────────────────────
# Everything the agent knows — passed between every node

class AgentState(TypedDict):
    # Data
    train_df:        Optional[pd.DataFrame]
    test_df:         Optional[pd.DataFrame]
    train_flags:     Optional[pd.DataFrame]   # LLM-extracted boolean flags
    test_flags:      Optional[pd.DataFrame]
    predictions:     Optional[pd.Series]

    # Control flow
    errors:          list[str]
    retry_count:     int
    llm_failed:      bool                     # triggers TF-IDF fallback
    current_node:    str

    # Config
    train_path:      str
    test_path:       str
    output_path:     str

# ── 2. Build the Graph ────────────────────────────────────────────────────────

def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("ingest",      ingest_node)
    graph.add_node("schema",      schema_repair_node)
    graph.add_node("llm_extract", llm_extract_node)
    graph.add_node("ml_train",    ml_train_node)
    graph.add_node("predict",     predict_node)
    graph.add_node("output",      output_node)

    # Entry point
    graph.set_entry_point("ingest")

    # Linear edges (happy path)
    graph.add_edge("ingest",   "schema")
    graph.add_edge("ml_train", "predict")
    graph.add_edge("predict",  "output")
    graph.add_edge("output",   END)

    # Conditional edge: after schema repair → LLM or retry?
    graph.add_conditional_edges(
        "schema",
        route_after_schema,
        {
            "llm_extract": "llm_extract",
            "ingest":      "ingest",       # retry if data completely broken
        }
    )

    # Conditional edge: after LLM extraction → train or fallback?
    graph.add_conditional_edges(
        "llm_extract",
        route_after_llm,
        {
            "ml_train":    "ml_train",     # LLM worked
            "ml_train_fb": "ml_train",     # LLM failed → TF-IDF fallback
                                           # same node, state.llm_failed=True
        }
    )

    return graph.compile()


# ── 3. Routing Functions ──────────────────────────────────────────────────────

def route_after_schema(state: AgentState) -> str:
    if state["train_df"] is None and state["retry_count"] < 3:
        return "ingest"   # something went wrong — retry
    return "llm_extract"

def route_after_llm(state: AgentState) -> str:
    if state["llm_failed"]:
        return "ml_train_fb"   # use TF-IDF instead
    return "ml_train"