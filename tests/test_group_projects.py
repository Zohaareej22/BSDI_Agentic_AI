from src.tools.project_tools import group_projects


print("\nEDUCATION PROJECTS BY DISTRICT")

result = group_projects(
    group_by="district",
    category="education",
)

print(result)


# ==================================================
# FIND ALL TOP DISTRICTS
# ==================================================

if result["results"]:

    highest_count = result["results"][0]["count"]

    top_districts = [
        item
        for item in result["results"]
        if item["count"] == highest_count
    ]

    print("\nTOP DISTRICT(S)")

    for item in top_districts:
        print(
            f"{item['group']}: "
            f"{item['count']} projects"
        )