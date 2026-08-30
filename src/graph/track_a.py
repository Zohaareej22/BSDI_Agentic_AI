import os
import json

from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from src.tools.project_tools import (
    query_projects,
    group_projects,
    rank_projects,
    filter_projects,
)


# ==================================================
# 1. CONNECT TO LOCAL QWEN MODEL
# ==================================================

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "qwen3:4b",
)

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    reasoning=False,
    disable_streaming=True,
    num_predict=300,
)


# ==================================================
# 2. CREATE LANGCHAIN TOOLS
# ==================================================

query_tool = StructuredTool.from_function(
    func=query_projects,
    name="query_projects",
    description=(
        "Query the BSDI infrastructure project dataset. "
        "Use for counting projects and calculating total "
        "or average project cost."
    ),
)

group_tool = StructuredTool.from_function(
    func=group_projects,
    name="group_projects",
    description=(
        "Group BSDI projects and count them by district, "
        "category, or status."
    ),
)

rank_tool = StructuredTool.from_function(
    func=rank_projects,
    name="rank_projects",
    description=(
        "Rank BSDI projects by cost. "
        "Use desc for most expensive and asc for least expensive."
    ),
)

filter_tool = StructuredTool.from_function(
    func=filter_projects,
    name="filter_projects",
    description=(
        "Return detailed BSDI project records matching "
        "district, category, and status filters."
    ),
)


tools = [
    query_tool,
    group_tool,
    rank_tool,
    filter_tool,
]


# ==================================================
# 3. QWEN
# ==================================================

llm_with_tools = llm.bind_tools(tools)


# ==================================================
# 4. AGENT NODE
# ==================================================

def agent(state: MessagesState):

    system_message = {
        "role": "system",
        "content": (
            "You are the BSDI infrastructure data assistant. "
            "Answer only using the supplied dataset information. "
            "Never invent numbers or project information. "
            "Be concise and direct. "
            "Do not reveal reasoning."
        ),
    }

    messages = [
        system_message
    ] + state["messages"]

    print("\n[AGENT] Calling Qwen...")

    response = llm.invoke(messages)

    print("[AGENT] Final response generated.")

    return {
        "messages": [response]
    }


# ==================================================
# 5. BUILD LANGGRAPH
# ==================================================

builder = StateGraph(MessagesState)

builder.add_node(
    "agent",
    agent,
)

builder.add_node(
    "tools",
    ToolNode(tools),
)

builder.add_edge(
    START,
    "agent",
)

builder.add_edge(
    "agent",
    "__end__",
)

graph = builder.compile()


# ==================================================
# 6. REASONING CLEANUP
# ==================================================

def _strip_reasoning(text: str) -> str:

    if not text:
        return ""

    text = str(text)

    if "</think>" in text:
        text = text.split(
            "</think>",
            1
        )[1]

    if "<think>" in text:
        text = text.split(
            "<think>",
            1
        )[0]

    return text.strip()


# ==================================================
# 7. DETERMINE TOOL FROM QUESTION
# ==================================================

def _select_tool(question: str):

    q = question.lower()

    # ----------------------------------------------
    # MOST / FEWEST BY DISTRICT
    # ----------------------------------------------

    if (
        ("district" in q)
        and (
            "most" in q
            or "highest" in q
            or "fewest" in q
            or "least" in q
        )
    ):

        return "group_projects"

    # ----------------------------------------------
    # MOST / FEWEST BY CATEGORY
    # ----------------------------------------------

    if (
        ("category" in q)
        and (
            "most" in q
            or "highest" in q
            or "fewest" in q
            or "least" in q
        )
    ):

        return "group_projects"

    # ----------------------------------------------
    # STATUS DISTRIBUTION
    # ----------------------------------------------

    if (
        "status" in q
        and (
            "most" in q
            or "fewest" in q
            or "distribution" in q
            or "how many" in q
        )
    ):

        return "group_projects"

    # ----------------------------------------------
    # MOST / LEAST EXPENSIVE
    # ----------------------------------------------

    if (
        "expensive" in q
        or "costliest" in q
        or "cheapest" in q
        or "least expensive" in q
    ):

        return "rank_projects"

    # ----------------------------------------------
    # TOTAL COST
    # ----------------------------------------------

    if (
        "total cost" in q
        or "total budget" in q
        or "overall cost" in q
    ):

        return "query_projects"

    # ----------------------------------------------
    # AVERAGE COST
    # ----------------------------------------------

    if (
        "average cost" in q
        or "avg cost" in q
        or "mean cost" in q
    ):

        return "query_projects"

    # ----------------------------------------------
    # COUNT
    # ----------------------------------------------

    if (
        "how many" in q
        or "number of projects" in q
        or "count" in q
    ):

        return "query_projects"

    # ----------------------------------------------
    # LIST / SHOW PROJECTS
    # ----------------------------------------------

    if (
        "show" in q
        or "list" in q
        or "which projects" in q
        or "projects in" in q
    ):

        return "filter_projects"

    return "query_projects"


# ==================================================
# 8. EXECUTE TOOL
# ==================================================

def _execute_tool(
    tool_name: str,
    question: str,
):

    q = question.lower()

    # ==================================================
    # GROUP PROJECTS
    # ==================================================

    if tool_name == "group_projects":

        group_by = "district"

        if (
            "by category" in q
            or "group by category" in q
            or "which category" in q
        ):
            group_by = "category"

        elif (
            "by status" in q
            or "group by status" in q
            or "which status" in q
        ):
            group_by = "status"

        category = None

        known_categories = [
            "education",
            "health",
            "water",
            "roads",
            "infrastructure",
            "agriculture",
        ]

        for value in known_categories:

            if value in q:

                category = value
                break

        try:

            if category:

                return group_projects(
                    group_by=group_by,
                    category=category,
                )

            return group_projects(
                group_by=group_by
            )

        except TypeError:

            return group_projects(
                group_by
            )

    # ==================================================
    # RANK PROJECTS
    # ==================================================

    if tool_name == "rank_projects":

        order = "desc"

        if (
            "cheapest" in q
            or "least expensive" in q
            or "least costly" in q
        ):

            order = "asc"

        try:

            return rank_projects(
                order=order
            )

        except TypeError:

            return rank_projects(
                order
            )

    # ==================================================
    # QUERY PROJECTS
    # ==================================================

    if tool_name == "query_projects":

        operation = "count"

        if (
            "total cost" in q
            or "total budget" in q
            or "overall cost" in q
        ):

            operation = "total_cost"

        elif (
            "average cost" in q
            or "avg cost" in q
            or "mean cost" in q
        ):

            operation = "average_cost"

        try:

            return query_projects(
                operation=operation
            )

        except TypeError:

            return query_projects(
                operation
            )

    # ==================================================
    # FILTER PROJECTS
    # ==================================================

    if tool_name == "filter_projects":

        return filter_projects()

    return None


# ==================================================
# 9. FINAL ANSWER GENERATION
# ==================================================

def _generate_final_answer(
    question: str,
    tool_name: str,
    tool_result,
):

    # For grouping questions, calculate the answer directly
    # from the tool result instead of asking Qwen to process
    # the entire dataset.
    if (
        tool_name == "group_projects"
        and isinstance(tool_result, dict)
        and "results" in tool_result
    ):

        results = tool_result["results"]

        if results:

            max_count = max(
                item["count"]
                for item in results
            )

            winners = [
                item["group"]
                for item in results
                if item["count"] == max_count
            ]

            if len(winners) == 1:

                return (
                    f"{winners[0]} has the most projects "
                    f"with {max_count} projects."
                )

            return (
                f"{' and '.join(winners)} have the most "
                f"projects with {max_count} projects each."
            )

    # For other Track A questions, keep Qwen as the
    # final answer generator.
    prompt = f"""
You are the BSDI infrastructure data assistant.

Answer the user's question using ONLY the dataset result below.

User question:
{question}

Dataset result:
{tool_result}

Rules:
- Give the direct answer.
- Use the actual numbers from the dataset.
- Do not invent information.
- Do not explain your reasoning.
- Do not mention tools.
- Do not say "let me analyze".
- Keep the answer concise.
"""

    response = llm.invoke(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    return _strip_reasoning(
        response.content
    )


# ==================================================
# 10. CALLABLE TRACK A FUNCTIONS
# ==================================================

def run_track_a(question):
    return run_track_a_traced(question)["answer"]


def run_track_a_traced(question: str) -> dict:
    tool_name = _select_tool(question)
    print("\n[AGENT] Calling Qwen...")
    print("[AGENT] Selected:", tool_name)
    tool_result = _execute_tool(tool_name, question)
    trace = [{"tool": tool_name, "arguments": {"question": question}, "result": tool_result}]
    answer = _generate_final_answer(question, tool_name, tool_result)
    print("[AGENT] Final response generated.")
    return {"question": question, "answer": answer, "trace": trace, "steps": len(trace)}


# ==================================================
# 11. TERMINAL MODE
# ==================================================

if __name__ == "__main__":

    question = input(
        "\nAsk the BSDI Agent: "
    )

    print("\n")
    print("=" * 50)
    print("TRACK A — BSDI DATA AGENT")
    print("=" * 50)

    answer = run_track_a(
        question
    )

    print("\n")
    print("=" * 50)
    print("FINAL AGENT RESPONSE")
    print("=" * 50)

    print(answer)