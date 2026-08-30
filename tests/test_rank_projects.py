from src.tools.project_tools import rank_projects


print("\nTOP 5 MOST EXPENSIVE HEALTH PROJECTS")

result = rank_projects(
    category="health",
    limit=5,
    order="desc",
)

for index, project in enumerate(
    result["results"],
    start=1,
):

    print(
        f"\n{index}. "
        f"{project['description']}"
    )

    print(
        f"   District: {project['district']}"
    )

    print(
        f"   Cost: {project['cost_m_pkr']} M PKR"
    )

    print(
        f"   Status: {project['status']}"
    )