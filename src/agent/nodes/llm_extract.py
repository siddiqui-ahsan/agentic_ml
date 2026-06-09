# agent/nodes/llm_extract.py

from agent.graph import AgentState
from agent.llm_feature_extractor import extract_flags_batch

def llm_extract_node(state: AgentState) -> AgentState:
    print("▶ Node: llm_extract")

    try:
        train_flags = extract_flags_batch(
            state["train_df"], 
            model="llama3.1:8b"   # your installed model
        )
        test_flags = extract_flags_batch(
            state["test_df"],
            model="llama3.1:8b"
        )
        return {**state,
            "train_flags":  train_flags,
            "test_flags":   test_flags,
            "llm_failed":   False,
            "current_node": "llm_extract",
        }

    except Exception as e:
        print(f"  LLM extraction failed: {e} — switching to TF-IDF fallback")
        return {**state,
            "llm_failed":   True,
            "errors":       state["errors"] + [f"LLM failed: {e}"],
            "current_node": "llm_extract",
        }