import os

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, MessagesState, START

from src.tools.audit_tools import run_audit_check


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "llama-3.1-8b-instant",
)

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)

# ==================================================
# 2. AUDIT TOOL
# ==================================================

audit_tool = StructuredTool.from_function(
    func=run_audit_check,
    name="run_audit_check",
    description=(
        "Run a BSDI portfolio audit check. "
        "Available checks: "
        "missing_work_started, "
        "high_cost_without_contractor, "
        "district_not_started_budget, "
        "category_cost_outliers, "
        "nits_no_but_in_progress."
    ),
)


# ==================================================
# 3. PLANNER
# ==================================================

def planner(state: MessagesState):

    last_message = state["messages"][-1]

    # Accept either a LangChain message object (.content) or a plain
    # {"role": ..., "content": ...} dict — the graph auto-converts dicts
    # to message objects, but run_track_b_detailed() calls this function
    # directly (to capture the plan text for the UI trace), so it won't
    # have gone through that conversion.
    user_goal = (
        last_message.content
        if hasattr(last_message, "content")
        else last_message["content"]
    )

    planner_prompt = f"""
You are the BSDI audit planning agent.

USER GOAL:
{user_goal}

AVAILABLE CHECKS:

missing_work_started
- In Progress projects with no Work Started date.

high_cost_without_contractor
- Top 10% cost projects with no contractor.

district_not_started_budget
- Districts where at least 50% of budget is Not Started.

category_cost_outliers
- Projects unusually expensive compared with their category.

nits_no_but_in_progress
- Projects where NITs = No and Status = In Progress.

Select all checks relevant to the user's goal.

For a broad portfolio-risk goal, select multiple relevant checks.

IMPORTANT:
Return ONLY the check names.
Do not explain.
Do not reason.
Do not write an essay.

Example:

missing_work_started
high_cost_without_contractor
category_cost_outliers
"""

    print(
        "\n[AGENT] Creating autonomous audit plan..."
    )

    response = llm.invoke(
        [
            {
                "role": "user",
                "content": planner_prompt,
            }
        ]
    )

    return {
        "messages": [response]
    }


# ==================================================
# 4. EXTRACT CHECKS
# ==================================================

def extract_checks(plan_text):

    available_checks = [
        "missing_work_started",
        "high_cost_without_contractor",
        "district_not_started_budget",
        "category_cost_outliers",
        "nits_no_but_in_progress",
    ]

    # Remove Qwen reasoning if present
    if "</think>" in plan_text:

        plan_text = plan_text.split(
            "</think>",
            1
        )[1]

    text = plan_text.lower()

    selected = []

    for check in available_checks:

        if check in text:

            if check not in selected:
                selected.append(check)

    return selected


# ==================================================
# 5. RUN AUDITS
# ==================================================

def run_selected_audits(checks):

    results = []

    print(
        "\n[AGENT] Running audit checks..."
    )

    for check in checks:

        print(
            f"[AGENT] Running: {check}"
        )

        result = run_audit_check(
            check
        )

        results.append(result)

        print(
            "[AGENT] Findings: "
            f"{result.get('count', 0)}"
        )

    return results


# ==================================================
# 6. RISK PRIORITY
# ==================================================

def calculate_priority(result):

    check = result.get(
        "check",
        ""
    )

    count = result.get(
        "count",
        0
    )

    # Higher priority for direct
    # project execution problems.
    priority_weights = {

        "missing_work_started": 100,

        "high_cost_without_contractor": 95,

        "category_cost_outliers": 80,

        "nits_no_but_in_progress": 75,

        "district_not_started_budget": 70,
    }

    base = priority_weights.get(
        check,
        50
    )

    # Count contributes slightly,
    # but does not dominate severity.
    count_score = min(
        count / 10,
        50
    )

    return base + count_score


# ==================================================
# 7. RISK LEVEL
# ==================================================

def risk_level(result):

    check = result.get(
        "check",
        ""
    )

    count = result.get(
        "count",
        0
    )

    if check in {
        "missing_work_started",
        "high_cost_without_contractor",
    }:
        return "HIGH"

    if check == "category_cost_outliers":

        if count >= 300:
            return "HIGH"

        return "MEDIUM"

    if check == "district_not_started_budget":

        if count >= 5:
            return "HIGH"

        return "MEDIUM"

    if check == "nits_no_but_in_progress":

        if count >= 150:
            return "HIGH"

        return "MEDIUM"

    return "MEDIUM"


# ==================================================
# 8. CREATE REPORT
# ==================================================

def create_risk_report(results):

    if not results:

        return (
            "No audit risks were identified."
        )

    # Rank results
    ranked = sorted(
        results,
        key=calculate_priority,
        reverse=True,
    )

    report = []

    report.append(
        "RANKED BSDI RISK REPORT"
    )

    report.append("")

    # --------------------------------------------------
    # Main risks
    # --------------------------------------------------

    for index, result in enumerate(
        ranked,
        start=1
    ):

        check = result.get(
            "check",
            "Unknown"
        )

        count = result.get(
            "count",
            0
        )

        level = risk_level(
            result
        )

        description = result.get(
            "description",
            ""
        )

        findings = result.get(
            "findings",
            []
        )

        report.append(
            f"{index}. {check}"
        )

        report.append(
            f"   Risk Level: {level}"
        )

        report.append(
            f"   Affected: {count}"
        )

        report.append(
            f"   Issue: {description}"
        )

        # --------------------------------------------------
        # Example
        # --------------------------------------------------

        if findings:

            example = findings[0]

            report.append(
                f"   Example: {example}"
            )

        report.append("")

    # --------------------------------------------------
    # Recommendations
    # --------------------------------------------------

    report.append(
        "RECOMMENDED ATTENTION"
    )

    report.append("")

    for index, result in enumerate(
        ranked[:3],
        start=1
    ):

        check = result.get(
            "check",
            ""
        )

        count = result.get(
            "count",
            0
        )

        recommendation_map = {

            "missing_work_started":
                f"Review {count} In Progress projects "
                "with missing Work Started dates.",

            "high_cost_without_contractor":
                f"Investigate {count} high-cost projects "
                "without assigned contractors.",

            "district_not_started_budget":
                f"Investigate the {count} districts "
                "with a high share of Not Started budget.",

            "category_cost_outliers":
                f"Review {count} projects with "
                "unusually high category costs.",

            "nits_no_but_in_progress":
                f"Review {count} projects where "
                "NITs is No but status is In Progress.",
        }

        recommendation = recommendation_map.get(
            check,
            f"Investigate {count} affected records."
        )

        report.append(
            f"{index}. {recommendation}"
        )

    return "\n".join(
        report
    )


# ==================================================
# 9. EXECUTOR
# ==================================================

def executor(state: MessagesState):

    plan = state["messages"][-1].content

    checks = extract_checks(
        plan
    )

    print("\n")
    print("=" * 60)
    print("AUTONOMOUS AUDIT PLAN")
    print("=" * 60)

    print(plan)

    print(
        "\n[AGENT] Selected checks:"
    )

    if not checks:

        print(
            "No valid checks selected."
        )

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "No valid audit checks "
                        "were selected."
                    ),
                }
            ]
        }

    for check in checks:

        print(
            f"  - {check}"
        )

    results = run_selected_audits(
        checks
    )

    print(
        "\n[AGENT] Creating risk ranking..."
    )

    report = create_risk_report(
        results
    )

    return {
        "messages": [
            {
                "role": "assistant",
                "content": report,
            }
        ]
    }


# ==================================================
# 10. BUILD GRAPH
# ==================================================

builder = StateGraph(
    MessagesState
)

builder.add_node(
    "planner",
    planner
)

builder.add_node(
    "executor",
    executor
)

builder.add_edge(
    START,
    "planner"
)

builder.add_edge(
    "planner",
    "executor"
)


# ==================================================
# 11. COMPILE
# ==================================================

graph = builder.compile()


# ==================================================
# 12. RUN
# ==================================================
# ==================================================
# RUN TRACK B — CALLABLE FUNCTION
# ==================================================

def run_track_b(goal):
    """
    Run Track B programmatically (report text only).

    Can be called from:
    - Streamlit UI
    - FastAPI
    - Other Python modules
    """
    return run_track_b_detailed(goal)["report"]


def run_track_b_detailed(goal: str) -> dict:
    """
    Run Track B and return the FULL audit trail:
    - the raw planning goal
    - the checks the agent decided to run (self-generated, not hardcoded)
    - the full findings for every check it ran (rows, counts, thresholds)
    - the final ranked, human-readable report

    This is what makes the plan -> act -> observe loop visible and
    auditable, per the assignment's transparency requirement.
    """

    plan_response = planner({"messages": [{"role": "user", "content": goal}]})
    plan_text = plan_response["messages"][0].content

    checks = extract_checks(plan_text)

    if not checks:
        return {
            "goal": goal,
            "plan_text": plan_text,
            "selected_checks": [],
            "results": [],
            "report": "No valid audit checks were selected.",
        }

    results = run_selected_audits(checks)

    ranked = sorted(results, key=calculate_priority, reverse=True)

    for result in ranked:
        result["risk_level"] = risk_level(result)
        result["priority_score"] = round(calculate_priority(result), 1)

    report = create_risk_report(results)

    return {
        "goal": goal,
        "plan_text": _strip_reasoning(plan_text),
        "selected_checks": checks,
        "results": ranked,
        "report": report,
    }


def _strip_reasoning(text: str) -> str:
    if text and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return (text or "").strip()


# ==================================================
# TERMINAL TEST MODE
# ==================================================

if __name__ == "__main__":

    goal = input(
        "\nEnter BSDI Audit Goal: "
    )

    print("\n")
    print("=" * 60)
    print("TRACK B — AUDIT AGENT")
    print("=" * 60)

    answer = run_track_b(goal)

    print("\n")
    print("=" * 60)
    print("RANKED BSDI RISK REPORT")
    print("=" * 60)

    print(answer)
