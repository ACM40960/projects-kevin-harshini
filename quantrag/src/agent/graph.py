"""
Phase 3 — LangGraph state graph wiring the 4 nodes together.
"""

from langgraph.graph import StateGraph, END
from src.agent.nodes import (
    AgentState,
    node_retriever,
    node_view_extractor,
    node_optimizer,
    node_report_generator,
)


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", node_retriever)
    graph.add_node("view_extractor", node_view_extractor)
    graph.add_node("optimizer", node_optimizer)
    graph.add_node("report_generator", node_report_generator)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "view_extractor")
    graph.add_edge("view_extractor", "optimizer")
    graph.add_edge("optimizer", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile()