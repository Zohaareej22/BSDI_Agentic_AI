import json
import requests

from src.tools.project_tools import (
    filter_projects,
    aggregate_projects,
)


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"


def call_llm(messages, tools):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["message"]


# ==================================================
# TOOL DEFINITIONS
# ==================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "filter_projects",
            "description": (
                "Filter BSDI projects by district, category, "
                "and status. Use this when project data needs "
                "to be selected before calculating an answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "Project district",
                    },
                    "category": {
                        "type": "string",
                        "description": "Project category",
                    },
                    "status": {
                        "type": "string",
                        "description": "Project status",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_projects",
            "description": (
                "Calculate a statistic from project rows returned "
                "by filter_projects. Supported operations are "
                "count, total_cost, and average_cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "description": (
                            "Project rows returned by filter_projects."
                        ),
                    },
                    "operation": {
                        "type": "string",
                        "enum": [
                            "count",
                            "total_cost",
                            "average_cost",
                        ],
                        "description": (
                            "Calculation to perform."
                        ),
                    },
                },
                "required": [
                    "rows",
                    "operation",
                ],
            },
        },
    },
]


# ==================================================
# CONVERSATION
# ==================================================

messages = [
    {
        "role": "system",
        "content": (
            "You are the BSDI infrastructure data assistant. "
            "Never guess project data. "
            "Use the provided tools whenever data is required. "
            "For questions requiring a calculation, use the "
            "appropriate tools and then provide a concise answer."
        ),
    },
    {
        "role": "user",
        "content": (
            "What is the total budget of all projects "
            "that have not started yet?"
        ),
    },
]


# ==================================================
# AGENT LOOP
# ==================================================

for step in range(5):

    print(f"\n[STEP {step + 1}] Asking Qwen...")

    message = call_llm(
        messages,
        tools,
    )

    tool_calls = message.get("tool_calls", [])

    # --------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------

    if not tool_calls:

        print("\n[FINAL ANSWER]")
        print(message.get("content", ""))

        break

    # --------------------------------------------------
    # QWEN REQUESTED TOOLS
    # --------------------------------------------------

    print("\n[TOOL CALLS]")

    for tool_call in tool_calls:

        function = tool_call["function"]

        name = function["name"]
        arguments = function["arguments"]

        print("Tool:", name)
        print("Arguments:", arguments)

        # ==============================================
        # FILTER PROJECTS
        # ==============================================

        if name == "filter_projects":

            rows = filter_projects(
                district=arguments.get("district"),
                category=arguments.get("category"),
                status=arguments.get("status"),
            )

            result = {
                "matching_rows": len(rows),
                "rows": rows,
            }

            print(
                "Filter result:",
                len(rows),
                "rows"
            )

        # ==============================================
        # AGGREGATE PROJECTS
        # ==============================================

        elif name == "aggregate_projects":

            rows = arguments.get("rows", [])

            operation = arguments.get(
                "operation"
            )

            result = aggregate_projects(
                rows,
                operation,
            )

            print(
                "Aggregation result:",
                result
            )

        else:

            result = {
                "error": f"Unknown tool: {name}"
            }

        # ==============================================
        # ADD TOOL CALL TO CONVERSATION
        # ==============================================

        messages.append(
            {
                "role": "assistant",
                "content": message.get(
                    "content",
                    ""
                ),
                "tool_calls": tool_calls,
            }
        )

        # ==============================================
        # ADD TOOL RESULT
        # ==============================================

        messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    result,
                    default=str,
                ),
            }
        )