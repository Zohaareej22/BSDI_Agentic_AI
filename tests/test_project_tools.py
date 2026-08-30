from src.tools.project_tools import (
    filter_projects,
    aggregate_projects,
)


# Test natural-language "water" → PHE
rows = filter_projects(
    district="Kech",
    category="water",
    status="completed",
)

print("Water projects in Kech that are completed:", len(rows))

count = aggregate_projects(rows, "count")
print("Count:", count)

total_cost = aggregate_projects(rows, "total_cost")
print("Total cost (M PKR):", total_cost)