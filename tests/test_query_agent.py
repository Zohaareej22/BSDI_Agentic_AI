import json
import requests

from src.tools.project_tools import query_projects


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
# TOOL DEFINITION
# ==================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_projects",
            "description": (
                "Query the BSDI infrastructure project dataset. "
                "You can filter by district, category, and status "
                "and perform count, total_cost, or average_cost. "
                "Use this tool instead of guessing project data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "Optional project district.",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional project category. "
                            "Use 'water' for water projects."
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
                        "description": "Calculation to perform.",
                    },
                },
                "required": [
                    "operation"
                ],
            },
        },
    }
]


# ==================================================
# USER QUESTION
# ==================================================

messages = [
    {
        "role": "system",
        "content": (
            "You are the BSDI infrastructure data assistant. "
            "Never guess numbers. "
            "When project data is needed, use query_projects. "
            "After receiving the tool result, give a concise "
            "final answer to the user."
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
# STEP 1 — QWEN
# ==================================================

print("\n[1] Asking Qwen...")

message = call_llm(
    messages,
    tools,
)

tool_calls = message.get(
    "tool_calls",
    []
)


# ==================================================
# STEP 2 — TOOL CALL
# ==================================================

if not tool_calls:

    print("\n❌ Qwen did not call query_projects.")
    print(message.get("content", ""))

    raise SystemExit(1)


tool_call = tool_calls[0]

function = tool_call["function"]

name = function["name"]
arguments = function["arguments"]


print("\n[2] Tool requested:")
print("Tool:", name)
print("Arguments:", arguments)


# ==================================================
# STEP 3 — EXECUTE REAL TOOL
# ==================================================

if name != "query_projects":

    raise ValueError(
        f"Unexpected tool: {name}"
    )


result = query_projects(
    district=arguments.get("district"),
    category=arguments.get("category"),
    status=arguments.get("status"),
    operation=arguments.get("operation", "count"),
)


print("\n[3] Tool result:")
print(result)


# ==================================================
# STEP 4 — SEND RESULT BACK TO QWEN
# ==================================================

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

messages.append(
    {
        "role": "tool",
        "content": json.dumps(
            result
        ),
    }
)


print("\n[4] Asking Qwen for final answer...")


final_message = call_llm(
    messages,
    [],
)


# ==================================================
# STEP 5 — FINAL ANSWER
# ==================================================

print("\n" + "=" * 50)
print("FINAL ANSWER")
print("=" * 50)

print(
    final_message.get(
        "content",
        ""
    )
)