# src/agent/graph.py

from langgraph.graph import StateGraph, END

from src.agent.state             import AgentState
from src.agent.nodes.ingest      import ingest_node
from src.agent.nodes.schema      import schema_repair_node
from src.agent.nodes.llm_extract import llm_extract_node
from src.agent.nodes.ml_train    import ml_train_node
from src.agent.nodes.predict     import predict_node
from src.agent.nodes.output      import output_node


def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("ingest",      ingest_node)
    graph.add_node("schema",      schema_repair_node)
    graph.add_node("llm_extract", llm_extract_node)
    graph.add_node("ml_train",    ml_train_node)
    graph.add_node("predict",     predict_node)
    graph.add_node("output",      output_node)

    graph.set_entry_point("ingest")

    graph.add_edge("ingest",   "schema")
    graph.add_edge("ml_train", "predict")
    graph.add_edge("predict",  "output")
    graph.add_edge("output",   END)

    graph.add_conditional_edges(
        "schema",
        route_after_schema,
        {
            "llm_extract": "llm_extract",
            "ingest":      "ingest",
        }
    )

    graph.add_conditional_edges(
        "llm_extract",
        route_after_llm,
        {
            "ml_train":    "ml_train",
            "ml_train_fb": "ml_train",
        }
    )

    return graph.compile()


def route_after_schema(state: AgentState) -> str:
    if state["train_df"] is None and state["retry_count"] < 3:
        return "ingest"
    return "llm_extract"

def route_after_llm(state: AgentState) -> str:
    if state["llm_failed"]:
        return "ml_train_fb"
    return "ml_train"
