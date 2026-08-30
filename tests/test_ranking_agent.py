import json
import requests

from src.tools.project_tools import rank_projects


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"


tools = [
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
                        "enum": ["asc", "desc"],
                    },
                },
            },
        },
    }
]


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
        timeout=180,
    )

    response.raise_for_status()

    return response.json()["message"]


messages = [
    {
        "role": "system",
        "content": (
            "You are the BSDI infrastructure data assistant. "
            "Never guess project data. "
            "Use the provided tool when project data is required. "
            "After receiving the tool result, give ONLY a concise "
            "answer to the user. Do not explain your reasoning."
        ),
    },
    {
        "role": "user",
        "content": (
            "What are the 5 most expensive health projects?"
        ),
    },
]


print("\n[1] Asking Qwen...")

message = call_llm(
    messages,
    tools,
)

tool_calls = message.get("tool_calls", [])

if not tool_calls:
    print("\nQwen did not call the tool.")
    print(message.get("content", ""))
    raise SystemExit


tool_call = tool_calls[0]

function = tool_call["function"]

name = function["name"]
arguments = function["arguments"]

print("\n[2] TOOL REQUEST")
print("Tool:", name)
print("Arguments:", arguments)


result = rank_projects(
    category=arguments.get("category"),
    status=arguments.get("status"),
    district=arguments.get("district"),
    limit=arguments.get("limit", 5),
    order=arguments.get("order", "desc"),
)


print("\n[3] TOOL RESULT")

print(
    json.dumps(
        result,
        indent=2,
        default=str,
    )
)


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

messages.append(
    {
        "role": "tool",
        "content": json.dumps(
            result,
            default=str,
        ),
    }
)


print("\n[4] Asking Qwen for final answer...")


final_message = call_llm(
    messages,
    [],
)


print("\n" + "=" * 50)
print("FINAL ANSWER")
print("=" * 50)

print(
    final_message.get(
        "content",
        "",
    )
)