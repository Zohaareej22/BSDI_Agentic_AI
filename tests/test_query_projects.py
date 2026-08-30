from src.tools.project_tools import query_projects


print("\nTEST 1")
print(
    query_projects(
        district="Kech",
        category="water",
        status="completed",
        operation="count",
    )
)


print("\nTEST 2")
print(
    query_projects(
        status="not started",
        operation="total_cost",
    )
)


print("\nTEST 3")
print(
    query_projects(
        category="health",
        operation="average_cost",
    )
)