from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    reasoning=False,
    num_predict=128,
)

response = llm.invoke(
    "Reply with exactly: TEST"
)

print(response.content)