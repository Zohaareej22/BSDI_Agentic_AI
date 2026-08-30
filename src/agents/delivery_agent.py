from src.tools.review_tools import delivery_review


def run_delivery_agent():
    """
    Delivery Agent:
    Evaluates whether Not Started projects
    have the accountability information needed
    for execution.
    """

    data = delivery_review(limit=10)

    recommendations = []
    concerns = []

    for project in data.get(
        "top_candidates",
        []
    ):

        missing = project.get(
            "missing_accountability",
            []
        )

        if not missing:

            recommendations.append({
                "global_id": project["global_id"],
                "district": project["district"],
                "category": project["category"],
                "cost_m_pkr": project["cost_m_pkr"],
                "reason": (
                    "No missing accountability fields "
                    "were identified."
                ),
            })

        else:

            concerns.append({
                "global_id": project["global_id"],
                "district": project["district"],
                "missing": missing,
                "reason": (
                    "Important delivery/accountability "
                    "information is missing."
                ),
            })

    return {
        "agent": "Delivery Agent",
        "recommendations": recommendations,
        "concerns": concerns,
    }


if __name__ == "__main__":

    result = run_delivery_agent()

    print("\nDELIVERY AGENT")
    print(result)