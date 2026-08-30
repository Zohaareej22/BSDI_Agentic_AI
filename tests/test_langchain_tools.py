from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool

from src.tools.project_tools import filter_projects


# Create the tool
filter_tool = StructuredTool.from_function(
    func=filter_projects,
    name="filter_projects",
    description=(
        "Filter BSDI projects by district, category, "
        "and status. Use this tool when the user asks "
        "about specific projects."
    ),
)


# Create Qwen
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    reasoning=False,
    num_predict=128,
)


# Give Qwen the tool
llm_with_tools = llm.bind_tools(
    [filter_tool]
)


# Ask a question that clearly requires the tool
response = llm_with_tools.invoke(
    "How many completed water projects are in Kech? "
    "Use the filter_projects tool to find the data."
)


print("\nMODEL RESPONSE:")
print(response.content)

print("\nTOOL CALLS:")
print(response.tool_calls)