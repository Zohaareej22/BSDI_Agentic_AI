import json
import requests

from src.tools.project_tools import (
    query_projects,
    group_projects,
    rank_projects,
    filter_projects,
)


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"


# ==================================================
# OLLAMA
# ==================================================

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

    # ------------------------------------------------
    # QUERY
    # ------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "query_projects",
            "description": (
                "Query BSDI project data. "
                "Use for counting projects or calculating "
                "total or average project cost. "
                "Never guess project numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "district": {
                        "type": "string",
                        "description": "Optional district.",
                    },

                    "category": {
                        "type": "string",
                        "description": (
                            "Optional category. "
                            "Use water for water projects."
                        ),
                    },

                    "status": {
                        "type": "string",
                        "description": "Optional project status.",
                    },

                    "operation": {
                        "type": "string",
                        "enum": [
                            "count",
                            "total_cost",
                            "average_cost",
                        ],
                    },
                },
                "required": [
                    "operation"
                ],
            },
        },
    },

    # ------------------------------------------------
    # GROUP
    # ------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "group_projects",
            "description": (
                "Group BSDI projects and count them. "
                "Use this when the user asks which district, "
                "category, or status has the most or fewest projects."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "group_by": {
                        "type": "string",
                        "enum": [
                            "district",
                            "category",
                            "status",
                        ],
                    },

                    "category": {
                        "type": "string",
                    },

                    "status": {
                        "type": "string",
                    },
                },
                "required": [
                    "group_by"
                ],
            },
        },
    },

    # ------------------------------------------------
    # RANK
    # ------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "rank_projects",
            "description": (
                "Rank BSDI projects by cost. "
                "Use this when the user asks for the "
                "most expensive or least expensive projects."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "category": {
                        "type": "string",
                    },

                    "status": {
                        "type": "string",
                    },

                    "district": {
                        "type": "string",
                    },

                    "limit": {
                        "type": "integer",
                    },

                    "order": {
                        "type": "string",
                        "enum": [
                            "asc",
                            "desc",
                        ],
                    },
                },
            },
        },
    },

    # ------------------------------------------------
    # FILTER
    # ------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "filter_projects",
            "description": (
                "Return detailed BSDI project records "
                "matching district, category, and status."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "district": {
                        "type": "string",
                    },

                    "category": {
                        "type": "string",
                    },

                    "status": {
                        "type": "string",
                    },
                },
            },
        },
    },
]


# ==================================================
# TOOL EXECUTION
# ==================================================

def execute_tool(name, arguments):

    if name == "query_projects":

        return query_projects(
            district=arguments.get("district"),
            category=arguments.get("category"),
            status=arguments.get("status"),
            operation=arguments.get(
                "operation",
                "count",
            ),
        )

    if name == "group_projects":

        return group_projects(
            group_by=arguments["group_by"],
            category=arguments.get("category"),
            status=arguments.get("status"),
        )

    if name == "rank_projects":

        return rank_projects(
            category=arguments.get("category"),
            status=arguments.get("status"),
            district=arguments.get("district"),
            limit=arguments.get("limit", 5),
            order=arguments.get(
                "order",
                "desc",
            ),
        )

    if name == "filter_projects":

        return filter_projects(
            district=arguments.get("district"),
            category=arguments.get("category"),
            status=arguments.get("status"),
        )

    return {
        "error": f"Unknown tool: {name}"
    }


# ==================================================
# AGENT
# ==================================================

def run_agent(question):

    messages = [
        {
            "role": "system",
            "content": (
                "You are the BSDI infrastructure data assistant. "
                "Never guess project data. "
                "Always use the appropriate tool when the "
                "answer requires dataset information. "
                "After receiving the tool result, answer "
                "the user's question concisely. "
                "Costs are reported in million PKR (M PKR). "
                "If multiple groups are tied for first place, "
                "mention all tied groups."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    for step in range(5):

        print(
            f"\n[STEP {step + 1}] Asking Qwen..."
        )

        message = call_llm(
            messages,
            tools,
        )

        tool_calls = message.get(
            "tool_calls",
            [],
        )

        # --------------------------------------------
        # FINAL ANSWER
        # --------------------------------------------

        if not tool_calls:

            return message.get(
                "content",
                "",
            )

        # --------------------------------------------
        # PROCESS TOOL CALLS
        # --------------------------------------------

        messages.append(
            {
                "role": "assistant",
                "content": message.get(
                    "content",
                    "",
                ),
                "tool_calls": tool_calls,
            }
        )

        for tool_call in tool_calls:

            function = tool_call["function"]

            name = function["name"]
            arguments = function["arguments"]

            print(
                "\n[TOOL REQUEST]"
            )

            print(
                "Tool:",
                name,
            )

            print(
                "Arguments:",
                arguments,
            )

            result = execute_tool(
                name,
                arguments,
            )

            print(
                "\n[TOOL RESULT]"
            )

            # Don't dump huge filter results
            if name == "filter_projects":

                print(
                    "Matching rows:",
                    len(result),
                )

            else:

                print(
                    json.dumps(
                        result,
                        indent=2,
                        default=str,
                    )
                )

            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        result,
                        default=str,
                    ),
                }
            )

    return "Agent stopped after maximum tool steps."


# ==================================================
# TEST QUESTIONS
# ==================================================

questions = [
    "How many completed water projects are in Kech?",

    "What is the total budget of all projects that have not started yet?",

    "Which district has the most education projects?",

    "What are the 5 most expensive health projects?",
]


# ==================================================
# RUN TESTS
# ==================================================

for number, question in enumerate(
    questions,
    start=1,
):

    print("\n")
    print("=" * 60)
    print(f"TEST {number}")
    print("=" * 60)

    print(
        "\nQUESTION:",
        question,
    )

    answer = run_agent(question)

    print(
        "\nFINAL ANSWER:"
    )

    print(answer)