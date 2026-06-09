# agent/run.py

from src.agent.graph import build_agent, AgentState

def run_agent(train_path: str, test_path: str, output_path: str) -> str:
    agent = build_agent()

    initial_state: AgentState = {
        "train_df":    None,
        "test_df":     None,
        "train_flags": None,
        "test_flags":  None,
        "predictions": None,
        "errors":      [],
        "retry_count": 0,
        "llm_failed":  False,
        "current_node": "start",
        "train_path":  train_path,
        "test_path":   test_path,
        "output_path": output_path,
    }

    final_state = agent.invoke(initial_state)

    if final_state["errors"]:
        print(f"Completed with warnings: {final_state['errors']}")

    return final_state["output_path"]