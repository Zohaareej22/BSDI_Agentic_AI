from ollama import chat


def get_project_count() -> int:
    """Return the number of projects in the BSDI dataset."""
    return 4083


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_project_count",
            "description": "Get the total number of projects in the BSDI dataset.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
]


messages = [
    {
        "role": "user",
        "content": "How many projects are in the BSDI dataset?"
    }
]


response = chat(
    model="qwen3:4b",
    messages=messages,
    tools=tools,
)


print("MODEL RESPONSE:")
print(response)

print("\nTOOL CALLS:")
print(response.message.tool_calls)