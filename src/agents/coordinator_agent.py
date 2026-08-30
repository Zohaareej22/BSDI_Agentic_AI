# ============================================================
# BSDI AGENTIC AI
# TRACK C — MULTI-AGENT REVIEW BOARD
# ============================================================

from src.agents.finance_agent import run_finance_agent
from src.agents.delivery_agent import run_delivery_agent
from src.agents.equity_agent import run_equity_agent


# ============================================================
# CONFIGURATION
# ============================================================

FUNDING_LIMIT_M_PKR = 2000
MAX_PROJECTS = 5


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def clean_text(value, default=""):
    if value is None:
        return default

    return str(value).strip()


# ============================================================
# BUILD UNIFIED CANDIDATE POOL
# ============================================================

def build_candidate_pool(
    finance,
    delivery,
    equity
):
    """
    Merge Finance + Delivery + Equity evidence.

    IMPORTANT:
    Delivery Agent stores projects with missing
    accountability information inside:

        delivery["concerns"]

    NOT delivery["top_candidates"].

    Equity evidence can be district-level or
    category-level.
    """

    pool = {}

    # ========================================================
    # FINANCE
    # ========================================================

    for project in finance.get(
        "recommendations",
        []
    ):

        project_id = project.get(
            "global_id"
        )

        if not project_id:
            continue

        pool[project_id] = {
            "global_id": project_id,

            "district": project.get(
                "district"
            ),

            "category": project.get(
                "category"
            ),

            "description": project.get(
                "description",
                ""
            ),

            "cost_m_pkr": safe_float(
                project.get(
                    "cost_m_pkr"
                )
            ),

            "category_average_m_pkr":
                project.get(
                    "category_average_m_pkr"
                ),

            "financial_assessment":
                project.get(
                    "financial_assessment",
                    ""
                ),

            "finance_support": True,

            "delivery_warning": None,

            "equity_support": False,

            "equity_reason": None,
        }

    # ========================================================
    # DELIVERY
    # ========================================================
    #
    # THIS IS THE IMPORTANT FIX.
    #
    # Your Delivery Agent returns:
    #
    # {
    #     "agent": "Delivery Agent",
    #     "recommendations": [...],
    #     "concerns": [...]
    # }
    #
    # Missing accountability projects are in "concerns".
    # ========================================================

    for project in delivery.get(
        "concerns",
        []
    ):

        project_id = project.get(
            "global_id"
        )

        if not project_id:
            continue

        # ----------------------------------------------------
        # If Finance did not include this project, add it.
        # ----------------------------------------------------

        if project_id not in pool:

            pool[project_id] = {
                "global_id": project_id,

                "district": project.get(
                    "district",
                    "Unknown"
                ),

                "category": project.get(
                    "category",
                    "Unknown"
                ),

                "description": project.get(
                    "description",
                    ""
                ),

                "cost_m_pkr": safe_float(
                    project.get(
                        "cost_m_pkr",
                        0
                    )
                ),

                "category_average_m_pkr":
                    None,

                "financial_assessment":
                    "",

                "finance_support":
                    False,

                "delivery_warning":
                    None,

                "equity_support":
                    False,

                "equity_reason":
                    None,
            }

        missing = project.get(
            "missing",
            []
        )

        pool[project_id][
            "delivery_warning"
        ] = {
            "missing": missing,

            "reason": project.get(
                "reason",
                (
                    "Important delivery/accountability "
                    "information is missing."
                )
            ),
        }

    # ========================================================
    # ALSO CHECK DELIVERY RECOMMENDATIONS
    # ========================================================

    for project in delivery.get(
        "recommendations",
        []
    ):

        project_id = project.get(
            "global_id"
        )

        if not project_id:
            continue

        if project_id not in pool:

            pool[project_id] = {
                "global_id": project_id,

                "district": project.get(
                    "district"
                ),

                "category": project.get(
                    "category"
                ),

                "description": project.get(
                    "description",
                    ""
                ),

                "cost_m_pkr": safe_float(
                    project.get(
                        "cost_m_pkr",
                        0
                    )
                ),

                "category_average_m_pkr":
                    None,

                "financial_assessment":
                    "",

                "finance_support":
                    False,

                "delivery_warning":
                    None,

                "equity_support":
                    False,

                "equity_reason":
                    None,
            }

    # ========================================================
    # EQUITY DISTRICTS
    # ========================================================

    equity_districts = {}

    for item in equity.get(
        "recommendations",
        []
    ):

        if item.get(
            "type"
        ) != "district":
            continue

        district = clean_text(
            item.get(
                "district"
            )
        )

        if not district:
            continue

        equity_districts[district] = {
            "reason": item.get(
                "reason",
                (
                    "District has comparatively low "
                    "Not Started allocation."
                )
            ),

            "projects": item.get(
                "projects",
                0
            ),

            "budget_m_pkr": item.get(
                "budget_m_pkr",
                0
            ),
        }

    # ========================================================
    # EQUITY CATEGORIES
    # ========================================================

    equity_categories = {}

    for item in equity.get(
        "concerns",
        []
    ):

        if item.get(
            "type"
        ) != "category":
            continue

        category = clean_text(
            item.get(
                "category"
            )
        )

        if not category:
            continue

        equity_categories[category] = {
            "reason": item.get(
                "reason",
                (
                    "Category has comparatively low "
                    "Not Started allocation."
                )
            ),

            "projects": item.get(
                "projects",
                0
            ),

            "budget_m_pkr": item.get(
                "budget_m_pkr",
                0
            ),
        }

    # ========================================================
    # APPLY EQUITY EVIDENCE
    # ========================================================

    for project in pool.values():

        district = clean_text(
            project.get(
                "district"
            )
        )

        category = clean_text(
            project.get(
                "category"
            )
        )

        district_evidence = (
            equity_districts.get(
                district
            )
        )

        category_evidence = (
            equity_categories.get(
                category
            )
        )

        reasons = []

        if district_evidence:

            reasons.append(
                f"District {district}: "
                f"{district_evidence['reason']}"
            )

        if category_evidence:

            reasons.append(
                f"Category {category}: "
                f"{category_evidence['reason']}"
            )

        if reasons:

            project[
                "equity_support"
            ] = True

            project[
                "equity_reason"
            ] = " ".join(
                reasons
            )

    return (
        list(pool.values()),
        equity_districts,
        equity_categories
    )


# ============================================================
# SCORE CANDIDATE
# ============================================================

def score_candidate(project):
    """
    Transparent Board scoring.

    Finance support:
        +3

    Equity support:
        +2

    No delivery warning:
        +2

    Delivery warning:
        +1

    Low-cost project:
        +1
    """

    score = 0.0

    if project.get(
        "finance_support"
    ):
        score += 3.0

    if project.get(
        "equity_support"
    ):
        score += 2.0

    if project.get(
        "delivery_warning"
    ):
        score += 1.0
    else:
        score += 2.0

    cost = safe_float(
        project.get(
            "cost_m_pkr"
        )
    )

    if cost <= 2:
        score += 1.0

    elif cost <= 5:
        score += 0.5

    return score


# ============================================================
# SELECT BOARD CANDIDATES
# ============================================================

def select_board_candidates(
    candidates
):
    """
    Select up to five projects.

    A Delivery-warning project is deliberately included
    when available so the Board can evaluate implementation
    risk rather than hiding it.
    """

    scored = []

    for project in candidates:

        scored.append(
            (
                score_candidate(
                    project
                ),
                project
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    selected_ids = set()

    # ========================================================
    # FIRST: INCLUDE A DELIVERY WARNING
    # ========================================================

    delivery_projects = [
        item
        for item in scored
        if item[1].get(
            "delivery_warning"
        )
    ]

    if delivery_projects:

        # Prefer one that also has equity support.
        delivery_projects.sort(
            key=lambda item: (
                bool(
                    item[1].get(
                        "equity_support"
                    )
                ),
                item[0]
            ),
            reverse=True
        )

        project = delivery_projects[0][1]

        selected.append(
            project
        )

        selected_ids.add(
            project.get(
                "global_id"
            )
        )

    # ========================================================
    # THEN FILL REMAINING POSITIONS
    # ========================================================

    for score, project in scored:

        if len(selected) >= MAX_PROJECTS:
            break

        project_id = project.get(
            "global_id"
        )

        if project_id in selected_ids:
            continue

        selected.append(
            project
        )

        selected_ids.add(
            project_id
        )

    return selected


# ============================================================
# DETECT BOARD TRADE-OFFS
# ============================================================

def detect_tradeoffs(
    candidates,
    equity_districts,
    equity_categories
):
    """
    Detect actual tensions between agent evidence.

    1. Finance vs Delivery
       Finance supports the project while Delivery
       reports missing accountability information.

    2. Equity vs Delivery
       Equity supports the district/category while
       Delivery reports missing accountability information.

    A Finance + Equity agreement is NOT a conflict.
    """

    tradeoffs = []

    for project in candidates:

        project_id = project.get(
            "global_id",
            "Unknown"
        )

        finance_support = bool(
            project.get(
                "finance_support"
            )
        )

        delivery_warning = bool(
            project.get(
                "delivery_warning"
            )
        )

        equity_support = bool(
            project.get(
                "equity_support"
            )
        )

        district = clean_text(
            project.get(
                "district"
            )
        )

        category = clean_text(
            project.get(
                "category"
            )
        )

        # ====================================================
        # FINANCE VS DELIVERY
        # ====================================================

        if (
            finance_support
            and
            delivery_warning
        ):

            warning = project.get(
                "delivery_warning",
                {}
            )

            missing = warning.get(
                "missing",
                []
            )

            tradeoffs.append({
                "project":
                    project_id,

                "type":
                    "Finance vs Delivery",

                "finance":
                    project.get(
                        "financial_assessment",
                        "Finance identified this project as a candidate."
                    ),

                "delivery":
                    (
                        "Missing: "
                        +
                        ", ".join(
                            missing
                        )
                    ),

                "equity":
                    project.get(
                        "equity_reason",
                        "No direct equity support identified."
                    ),

                "resolution":
                    (
                        "Finance supports consideration, "
                        "but Delivery has identified "
                        "accountability gaps. The Board "
                        "places the project on HOLD / VERIFY "
                        "until the missing information is verified."
                    ),
            })

            continue

        # ====================================================
        # EQUITY VS DELIVERY
        # ====================================================

        district_equity = (
            district in equity_districts
        )

        category_equity = (
            category in equity_categories
        )

        if (
            delivery_warning
            and
            (
                equity_support
                or
                district_equity
                or
                category_equity
            )
        ):

            warning = project.get(
                "delivery_warning",
                {}
            )

            missing = warning.get(
                "missing",
                []
            )

            equity_reason = project.get(
                "equity_reason"
            )

            if not equity_reason:

                if district_equity:

                    equity_reason = (
                        f"Equity Agent identified "
                        f"{district} as a district with "
                        f"relatively low Not Started allocation."
                    )

                elif category_equity:

                    equity_reason = (
                        f"Equity Agent identified "
                        f"{category} as a category with "
                        f"relatively low Not Started allocation."
                    )

                else:

                    equity_reason = (
                        "Equity evidence supports consideration."
                    )

            tradeoffs.append({
                "project":
                    project_id,

                "type":
                    "Equity vs Delivery",

                "finance":
                    project.get(
                        "financial_assessment",
                        "Finance evidence was reviewed."
                    ),

                "delivery":
                    (
                        "Missing: "
                        +
                        ", ".join(
                            missing
                        )
                    ),

                "equity":
                    equity_reason,

                "resolution":
                    (
                        "The Board recognizes the equity "
                        "consideration but places the project "
                        "on HOLD / VERIFY because delivery "
                        "and accountability information is missing."
                    ),
            })

    return tradeoffs


# ============================================================
# PROJECT DECISION
# ============================================================

def make_project_decision(
    project
):
    """
    Delivery warning takes priority.

    FUND:
        Supporting Finance/Equity evidence and
        no Delivery warning.

    HOLD / VERIFY:
        Delivery/accountability warning exists.

    REVIEW:
        Insufficient supporting evidence.
    """

    delivery_warning = bool(
        project.get(
            "delivery_warning"
        )
    )

    finance_support = bool(
        project.get(
            "finance_support"
        )
    )

    equity_support = bool(
        project.get(
            "equity_support"
        )
    )

    if delivery_warning:

        return (
            "HOLD / VERIFY",
            (
                "Delivery/accountability information "
                "must be verified before funding."
            )
        )

    if (
        finance_support
        or
        equity_support
    ):

        return (
            "FUND",
            (
                "The project has supporting Finance "
                "and/or Equity evidence and no Delivery "
                "warning was identified."
            )
        )

    return (
        "REVIEW",
        (
            "The project requires further Board review "
            "because limited supporting evidence was identified."
        )
    )


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    selected,
    tradeoffs,
    total_funding
):

    remaining = (
        FUNDING_LIMIT_M_PKR
        -
        total_funding
    )

    lines = []

    # ========================================================
    # DECISION
    # ========================================================

    lines.append(
        "FINAL BOARD DECISION"
    )

    lines.append("")

    lines.append(
        "Recommended strategy: Balanced allocation "
        "based on Finance, Delivery and Equity evidence."
    )

    lines.append("")

    # ========================================================
    # AGENT POSITIONS
    # ========================================================

    lines.append(
        "AGENT POSITIONS"
    )

    lines.append("")

    lines.append(
        "Finance Agent:"
    )

    lines.append(
        "Evaluated Not Started projects using "
        "cost efficiency and category-level "
        "financial evidence."
    )

    lines.append("")

    lines.append(
        "Delivery Agent:"
    )

    lines.append(
        "Evaluated implementation and accountability "
        "readiness and identified missing information."
    )

    lines.append("")

    lines.append(
        "Equity Agent:"
    )

    lines.append(
        "Evaluated relatively low Not Started "
        "allocations across districts and categories."
    )

    lines.append("")

    # ========================================================
    # DELIBERATION
    # ========================================================

    lines.append(
        "BOARD DELIBERATION"
    )

    lines.append("")

    lines.append(
        "The Coordinator compared Finance, Delivery "
        "and Equity evidence and resolved identified "
        "trade-offs before producing the final ranking."
    )

    lines.append("")

    # ========================================================
    # TRADE-OFFS
    # ========================================================

    lines.append(
        "BOARD TRADE-OFFS"
    )

    lines.append("")

    if tradeoffs:

        for item in tradeoffs:

            lines.append(
                f"- {item['project']}: "
                f"{item['type']}"
            )

            lines.append(
                f"  Finance: "
                f"{item['finance']}"
            )

            lines.append(
                f"  Delivery: "
                f"{item['delivery']}"
            )

            lines.append(
                f"  Equity: "
                f"{item['equity']}"
            )

            lines.append(
                f"  Board Resolution: "
                f"{item['resolution']}"
            )

            lines.append("")

    else:

        lines.append(
            "No direct trade-off was detected "
            "among the selected Board candidates."
        )

        lines.append("")

    # ========================================================
    # RANKING
    # ========================================================

    lines.append(
        "RANKED FUNDING RECOMMENDATION"
    )

    lines.append("")

    for index, project in enumerate(
        selected,
        start=1
    ):

        project_id = project.get(
            "global_id",
            "Unknown"
        )

        district = project.get(
            "district",
            "Unknown"
        )

        category = project.get(
            "category",
            "Unknown"
        )

        cost = safe_float(
            project.get(
                "cost_m_pkr"
            )
        )

        decision, reason = make_project_decision(
            project
        )

        evidence = []

        if project.get(
            "finance_support"
        ):
            evidence.append(
                "Finance-supported"
            )

        if project.get(
            "equity_support"
        ):
            evidence.append(
                "Equity-supported"
            )

        if project.get(
            "delivery_warning"
        ):
            evidence.append(
                "Delivery warning"
            )
        else:
            evidence.append(
                "No delivery warning"
            )

        lines.append(
            f"{index}. {project_id}"
        )

        lines.append(
            f"   District: {district}"
        )

        lines.append(
            f"   Category: {category}"
        )

        lines.append(
            f"   Cost: {cost:g} M PKR"
        )

        lines.append(
            "   Status: Not Started"
        )

        lines.append(
            f"   Decision: {decision}"
        )

        lines.append(
            "   Evidence: "
            +
            "; ".join(
                evidence
            )
        )

        lines.append(
            f"   Reason: {reason}"
        )

        lines.append("")

    # ========================================================
    # ATTENTION
    # ========================================================

    lines.append(
        "RECOMMENDED ATTENTION"
    )

    lines.append("")

    lines.append(
        "1. Verify contractor and XEN information "
        "where Delivery identified missing fields."
    )

    lines.append(
        "2. Continue monitoring implementation readiness."
    )

    lines.append(
        "3. Consider district and category equity "
        "when allocating remaining funds."
    )

    lines.append("")

    # ========================================================
    # FUNDING
    # ========================================================

    lines.append(
        "FUNDING SUMMARY"
    )

    lines.append("")

    lines.append(
        f"Funding envelope: "
        f"{FUNDING_LIMIT_M_PKR:.4f} M PKR"
    )

    lines.append(
        f"Recommended funding: "
        f"{total_funding:.4f} M PKR"
    )

    lines.append(
        f"Remaining funding: "
        f"{remaining:.4f} M PKR"
    )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN COORDINATOR
# ============================================================

def run_coordinator():

    # ========================================================
    # FINANCE
    # ========================================================

    print(
        "[COORDINATOR] Running Finance Agent..."
    )

    finance = run_finance_agent()

    print(
        "[COORDINATOR] Finance Agent complete."
    )

    # ========================================================
    # DELIVERY
    # ========================================================

    print(
        "[COORDINATOR] Running Delivery Agent..."
    )

    delivery = run_delivery_agent()

    print(
        "[COORDINATOR] Delivery Agent complete."
    )

    # ========================================================
    # EQUITY
    # ========================================================

    print(
        "[COORDINATOR] Running Equity Agent..."
    )

    equity = run_equity_agent()

    print(
        "[COORDINATOR] Equity Agent complete."
    )

    # ========================================================
    # UNIFIED POOL
    # ========================================================

    print(
        "[COORDINATOR] Preparing unified candidate pool..."
    )

    (
        candidates,
        equity_districts,
        equity_categories
    ) = build_candidate_pool(
        finance,
        delivery,
        equity
    )

    print(
        f"[COORDINATOR] Unified pool: "
        f"{len(candidates)} candidates."
    )

    # ========================================================
    # BOARD SELECTION
    # ========================================================

    print(
        "[COORDINATOR] Selecting Board candidates..."
    )

    selected = select_board_candidates(
        candidates
    )

    print(
        f"[COORDINATOR] Selected "
        f"{len(selected)} Board candidates."
    )

    # ========================================================
    # TRADE-OFFS
    # ========================================================

    print(
        "[COORDINATOR] Detecting board trade-offs..."
    )

    tradeoffs = detect_tradeoffs(
        selected,
        equity_districts,
        equity_categories
    )

    print(
        f"[COORDINATOR] Detected "
        f"{len(tradeoffs)} "
        f"board trade-off(s)."
    )

    # ========================================================
    # FUNDING
    # ========================================================

    total_funding = 0.0

    for project in selected:

        decision, _ = make_project_decision(
            project
        )

        if decision == "FUND":

            total_funding += safe_float(
                project.get(
                    "cost_m_pkr"
                )
            )

    print(
        f"[COORDINATOR] Funding total: "
        f"{total_funding:.4f} M PKR"
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "[COORDINATOR] Generating board recommendation..."
    )

    recommendation = generate_report(
        selected,
        tradeoffs,
        total_funding
    )

    print(
        "[COORDINATOR] Board recommendation complete."
    )

    # ========================================================
    # RETURN STRUCTURE
    # ========================================================

    return {
        "recommendation":
            recommendation,

        "finance":
            finance,

        "delivery":
            delivery,

        "equity":
            equity,

        "selected":
            selected,

        "tradeoffs":
            tradeoffs,

        "total_funding":
            total_funding,

        "funding_limit":
            FUNDING_LIMIT_M_PKR,

        "remaining_funding":
            (
                FUNDING_LIMIT_M_PKR
                -
                total_funding
            ),
    }


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "TRACK C — MULTI-AGENT REVIEW BOARD"
    )
    print("=" * 60)
    print()

    result = run_coordinator()

    print()
    print("=" * 60)
    print(
        "FINAL BOARD RECOMMENDATION"
    )
    print("=" * 60)
    print()

    print(
        result["recommendation"]
    )