from src.tools.review_tools import equity_review


def run_equity_agent():
    """
    Equity Agent:
    Evaluates districts and categories with
    relatively low Not Started allocations.
    """

    data = equity_review(limit=10)

    recommendations = []
    concerns = []

    # --------------------------------------------------
    # District equity
    # --------------------------------------------------

    for district in data.get(
        "least_funded_districts",
        []
    ):

        recommendations.append({
            "type": "district",
            "district": district["district"],
            "projects": district[
                "not_started_projects"
            ],
            "budget_m_pkr": district[
                "not_started_budget_m_pkr"
            ],
            "reason": (
                "District has comparatively low "
                "Not Started allocation."
            ),
        })

    # --------------------------------------------------
    # Category equity
    # --------------------------------------------------

    for category in data.get(
        "least_funded_categories",
        []
    ):

        concerns.append({
            "type": "category",
            "category": category["category"],
            "projects": category[
                "not_started_projects"
            ],
            "budget_m_pkr": category[
                "not_started_budget_m_pkr"
            ],
            "reason": (
                "Category has comparatively low "
                "Not Started allocation."
            ),
        })

    return {
        "agent": "Equity Agent",
        "recommendations": recommendations,
        "concerns": concerns,
    }


if __name__ == "__main__":

    result = run_equity_agent()

    print("\nEQUITY AGENT")
    print(result)