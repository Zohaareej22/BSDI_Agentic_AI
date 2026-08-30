from src.tools.review_tools import finance_review


FUNDING_LIMIT_M_PKR = 2000
CANDIDATE_LIMIT = 50


def run_finance_agent():
    """
    Finance Agent:
    Evaluates Not Started projects from a financial perspective.

    The agent:
    - Reviews a broader candidate pool.
    - Identifies cost-efficient projects.
    - Flags unusually expensive candidates.
    - Checks the total unfunded pipeline against the
      PKR 2 billion funding envelope.
    """

    data = finance_review(
        limit=CANDIDATE_LIMIT
    )

    recommendations = []
    concerns = []

    # --------------------------------------------------
    # TOTAL NOT STARTED BUDGET
    # --------------------------------------------------

    total_budget = data.get(
        "total_not_started_budget_m_pkr",
        0
    )

    if total_budget > FUNDING_LIMIT_M_PKR:

        concerns.append({
            "issue": "Large unfunded pipeline",
            "evidence": (
                f"{total_budget} M PKR in Not Started projects "
                f"compared with a "
                f"{FUNDING_LIMIT_M_PKR} M PKR funding envelope."
            ),
        })

    # --------------------------------------------------
    # ANALYZE CANDIDATES
    # --------------------------------------------------

    candidates = data.get(
        "top_candidates",
        []
    )

    for project in candidates:

        cost = project.get(
            "cost_m_pkr",
            0
        )

        category_average = project.get(
            "category_average_m_pkr"
        )

        # ----------------------------------------------
        # Determine financial position
        # ----------------------------------------------

        if (
            category_average
            and category_average > 0
        ):

            cost_ratio = (
                cost / category_average
            )

        else:

            cost_ratio = 0

        # ----------------------------------------------
        # Financial assessment
        # ----------------------------------------------

        if cost_ratio <= 0.5:

            financial_assessment = (
                "Strong cost efficiency: "
                "project cost is substantially below "
                "the category average."
            )

        elif cost_ratio <= 1.0:

            financial_assessment = (
                "Cost-efficient candidate: "
                "project cost is at or below "
                "the category average."
            )

        elif cost_ratio <= 1.5:

            financial_assessment = (
                "Moderate financial efficiency: "
                "project cost is above the category average."
            )

        else:

            financial_assessment = (
                "Potential cost concern: "
                "project cost is substantially above "
                "the category average."
            )

        recommendation = {
            "global_id": project.get(
                "global_id"
            ),
            "district": project.get(
                "district"
            ),
            "category": project.get(
                "category"
            ),
            "description": project.get(
                "description"
            ),
            "cost_m_pkr": cost,
            "category_average_m_pkr": (
                category_average
            ),
            "financial_assessment": (
                financial_assessment
            ),
        }

        recommendations.append(
            recommendation
        )

        # ----------------------------------------------
        # Flag expensive candidates
        # ----------------------------------------------

        if cost_ratio > 1.5:

            concerns.append({
                "issue": "Category cost outlier",
                "global_id": project.get(
                    "global_id"
                ),
                "evidence": (
                    f"Cost {cost} M PKR is "
                    f"{round(cost_ratio, 2)}x "
                    f"the category average of "
                    f"{category_average} M PKR."
                ),
            })

    # --------------------------------------------------
    # RETURN STRUCTURED AGENT OUTPUT
    # --------------------------------------------------

    return {
        "agent": "Finance Agent",

        "recommendations": recommendations,

        "concerns": concerns,

        "evidence": {
            "total_not_started_budget_m_pkr": (
                total_budget
            ),
            "funding_limit_m_pkr": (
                FUNDING_LIMIT_M_PKR
            ),
            "candidate_pool_size": (
                len(recommendations)
            ),
        },
    }


# ==================================================
# TEST FINANCE AGENT
# ==================================================

if __name__ == "__main__":

    result = run_finance_agent()

    print("\n")
    print("=" * 60)
    print("FINANCE AGENT")
    print("=" * 60)

    print(result)