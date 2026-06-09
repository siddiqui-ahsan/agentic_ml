# src/agent/run.py

from src.agent.graph import build_agent
from src.agent.state import AgentState


def run_agent(train_path: str, test_path: str, output_path: str) -> str:
    agent = build_agent()

    initial_state: AgentState = {
        # Data
        "train_df":     None,
        "test_df":      None,
        "train_flags":  None,
        "test_flags":   None,
        "predictions":  None,
        "property_ids": None,
        # ML
        "pipeline":     None,
        "flag_cols":    [],
        # Control flow
        "errors":       [],
        "retry_count":  0,
        "llm_failed":   False,
        "current_node": "start",
        # Config
        "train_path":   train_path,
        "test_path":    test_path,
        "output_path":  output_path,
    }

    final_state = agent.invoke(initial_state)

    if final_state["errors"]:
        print(f"Completed with warnings: {final_state['errors']}")

    return final_state["output_path"]
