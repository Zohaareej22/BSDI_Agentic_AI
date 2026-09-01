"""
BSDI Agentic AI - Track A

Groq + LangGraph single-agent infrastructure data assistant.

The LLM interprets the user's natural-language question and creates
a structured query plan.

Pandas executes that plan against Projects.xlsx.

The LLM never invents the numeric result.
The dataset remains the source of truth.
"""

import json
import os
import re
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode

from src.tools.project_tools import (
    query_projects,
    group_projects,
    rank_projects,
    filter_projects,
    PROJECTS_DF,
)


# ============================================================
# 1. GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "",
)

GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


def _make_llm():

    if not GROQ_API_KEY:

        return None

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )


llm = _make_llm()


# ============================================================
# 2. DATASET SCHEMA
# ============================================================

DATASET_COLUMNS = [
    str(column)
    for column in PROJECTS_DF.columns
]


# ============================================================
# 3. LANGCHAIN TOOLS
# ============================================================

query_tool = StructuredTool.from_function(
    func=query_projects,
    name="query_projects",
    description=(
        "Query the BSDI infrastructure dataset. "
        "Supports count, sum, average, min, max, "
        "unique values, grouping, group sums, "
        "group averages, listing, and ranking."
    ),
)


group_tool = StructuredTool.from_function(
    func=group_projects,
    name="group_projects",
    description=(
        "Group BSDI projects by district, category, "
        "status, phase, agency, or another dataset field."
    ),
)


rank_tool = StructuredTool.from_function(
    func=rank_projects,
    name="rank_projects",
    description=(
        "Rank projects by cost and return the highest "
        "or lowest projects."
    ),
)


filter_tool = StructuredTool.from_function(
    func=filter_projects,
    name="filter_projects",
    description=(
        "List project records matching district, "
        "category, status, or advanced filters."
    ),
)


TOOLS = [
    query_tool,
    group_tool,
    rank_tool,
    filter_tool,
]


# ============================================================
# 4. HELPERS
# ============================================================

def _strip_reasoning(text: Any) -> str:

    if text is None:
        return ""

    text = str(text)

    if "</think>" in text:

        text = text.split(
            "</think>",
            1,
        )[1]

    if "<think>" in text:

        text = text.split(
            "<think>",
            1,
        )[0]

    return text.strip()


def _extract_json(text: str):

    text = _strip_reasoning(text)

    # Remove markdown fences
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        "",
    ).strip()

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        candidate = text[
            start:end + 1
        ]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def _extract_limit(
    question: str,
    default: int = 10,
) -> int:

    patterns = [
        r"\btop\s+(\d+)",
        r"\bfirst\s+(\d+)",
        r"\blast\s+(\d+)",
        r"\b(\d+)\s+(?:most|least)",
        r"\b(\d+)\s+projects?\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question.lower(),
        )

        if match:

            return max(
                1,
                min(
                    int(match.group(1)),
                    100,
                ),
            )

    return default


# ============================================================
# 5. VALUE EXTRACTION FALLBACK
# ============================================================

CATEGORY_ALIASES = {
    "water": "PHE",
    "phe": "PHE",
    "education": "Education",
    "health": "Health",
    "healthcare": "Health",
    "road": "Road",
    "roads": "Road",
    "irrigation": "Irrigation",
    "agriculture": "Agriculture",
    "energy": "Energy",
    "building": "Building",
    "municipal": "Municipal",
    "sewerage": "Sewerage",
    "security": "Security",
    "sports": "Sports",
}


STATUS_ALIASES = {
    "completed": "Completed",
    "complete": "Completed",
    "in progress": "In Progress",
    "ongoing": "In Progress",
    "not started": "Not Started",
    "nits issued": "NITs Issued",
}


def _fallback_category(question: str):

    q = question.lower()

    for alias in sorted(
        CATEGORY_ALIASES,
        key=len,
        reverse=True,
    ):

        if alias in q:

            return CATEGORY_ALIASES[
                alias
            ]

    return None


def _fallback_status(question: str):

    q = question.lower()

    for alias in sorted(
        STATUS_ALIASES,
        key=len,
        reverse=True,
    ):

        if alias in q:

            return STATUS_ALIASES[
                alias
            ]

    return None


def _fallback_district(question: str):

    q = question.lower()

    districts = (
        PROJECTS_DF["District"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    districts = sorted(
        districts,
        key=len,
        reverse=True,
    )

    for district in districts:

        if district.lower() in q:

            return district

    return None


# ============================================================
# 6. RULE-BASED FALLBACK PLANNER
# ============================================================

def _fallback_plan(
    question: str,
) -> dict:

    q = question.lower()

    category = _fallback_category(
        question
    )

    status = _fallback_status(
        question
    )

    district = _fallback_district(
        question
    )

    limit = _extract_limit(
        question
    )

    filters = []

    if district:

        filters.append(
            {
                "field": "District",
                "operator": "eq",
                "value": district,
            }
        )

    if category:

        filters.append(
            {
                "field": "Category",
                "operator": "eq",
                "value": category,
            }
        )

    if status:

        filters.append(
            {
                "field": "Status",
                "operator": "eq",
                "value": status,
            }
        )

    # --------------------------------------------------------
    # Operation detection
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "top ",
            "most expensive",
            "most costly",
            "highest cost",
            "highest-cost",
            "cheapest",
            "least expensive",
            "lowest cost",
            "lowest-cost",
        )
    ):

        operation = (
            "bottom"
            if any(
                phrase in q
                for phrase in (
                    "cheapest",
                    "least expensive",
                    "lowest",
                )
            )
            else "top"
        )

        return {
            "operation": operation,
            "field": "Cost (M)",
            "limit": limit,
            "order": (
                "asc"
                if operation == "bottom"
                else "desc"
            ),
            "filters": filters,
        }

    if any(
        word in q
        for word in (
            "list",
            "show",
            "display",
            "which projects",
            "give me the projects",
        )
    ):

        return {
            "operation": "list",
            "limit": limit,
            "filters": filters,
        }

    # Grouping
    if (
        "by district" in q
        or "each district" in q
        or "per district" in q
    ):

        if any(
            word in q
            for word in (
                "budget",
                "cost",
                "total",
                "sum",
            )
        ):

            operation = "group_sum"

            field = "Cost (M)"

        else:

            operation = "group"

            field = None

        return {
            "operation": operation,
            "group_by": "District",
            "field": field,
            "filters": filters,
        }

    if (
        "by category" in q
        or "each category" in q
        or "per category" in q
    ):

        if any(
            word in q
            for word in (
                "budget",
                "cost",
                "total",
                "sum",
            )
        ):

            operation = "group_sum"

            field = "Cost (M)"

        else:

            operation = "group"

            field = None

        return {
            "operation": operation,
            "group_by": "Category",
            "field": field,
            "filters": filters,
        }

    if (
        "by status" in q
        or "each status" in q
        or "per status" in q
    ):

        return {
            "operation": "group",
            "group_by": "Status",
            "filters": filters,
        }

    # Average
    if any(
        phrase in q
        for phrase in (
            "average",
            "avg",
            "mean",
        )
    ):

        return {
            "operation": "average",
            "field": "Cost (M)",
            "filters": filters,
        }

    # Maximum
    if any(
        phrase in q
        for phrase in (
            "maximum",
            "max",
            "highest budget",
            "highest cost",
        )
    ):

        return {
            "operation": "max",
            "field": "Cost (M)",
            "filters": filters,
        }

    # Minimum
    if any(
        phrase in q
        for phrase in (
            "minimum",
            "min",
            "lowest budget",
            "lowest cost",
        )
    ):

        return {
            "operation": "min",
            "field": "Cost (M)",
            "filters": filters,
        }

    # Sum / budget
    if any(
        phrase in q
        for phrase in (
            "total budget",
            "total cost",
            "overall budget",
            "overall cost",
            "sum of",
            "budget of",
            "cost of",
        )
    ):

        return {
            "operation": "sum",
            "field": "Cost (M)",
            "filters": filters,
        }

    # Default
    return {
        "operation": "count",
        "filters": filters,
    }


# ============================================================
# 7. GROQ PLANNER
# ============================================================

def _create_plan(
    question: str,
) -> dict:

    fallback = _fallback_plan(
        question
    )

    # Deterministic handling for missing contractors.
    # Do not allow the LLM to ignore this filter.
    q_lower = question.lower()

    if any(
        phrase in q_lower
        for phrase in (
            "without contractor",
            "without a contractor",
            "no contractor",
            "no contractor assigned",
            "contractor not assigned",
            "contractor is missing",
            "missing contractor",
        )
    ):
        return {
            "operation": "count",
            "filters": [
                {
                    "field": "Contractor",
                    "operator": "is_null",
                }
            ],
        }

    if llm is None:
        return fallback

    schema_text = "\n".join(
        f"- {column}"
        for column in DATASET_COLUMNS
    )

    prompt = f"""
You are the planning component of a BSDI infrastructure
data analysis agent.

The user asks a natural-language question about Projects.xlsx.

Your job is ONLY to convert the question into a JSON query plan.

Do NOT answer the question.

Dataset columns:
{schema_text}

Supported operations:

count
sum
average
min
max
unique
group
group_sum
group_average
list
top
bottom

The main numeric budget field is:
Cost (M)

Examples:

Question:
"What is the total budget of Khuzdar?"

Plan:
{{
  "operation": "sum",
  "field": "Cost (M)",
  "filters": [
    {{"field": "District", "operator": "eq", "value": "Khuzdar"}}
  ]
}}

Question:
"How many completed health projects are in Kech?"

Plan:
{{
  "operation": "count",
  "filters": [
    {{"field": "District", "operator": "eq", "value": "Kech"}},
    {{"field": "Category", "operator": "eq", "value": "Health"}},
    {{"field": "Status", "operator": "eq", "value": "Completed"}}
  ]
}}

Question:
"List the 5 most expensive health projects."

Plan:
{{
  "operation": "top",
  "field": "Cost (M)",
  "limit": 5,
  "order": "desc",
  "filters": [
    {{"field": "Category", "operator": "eq", "value": "Health"}}
  ]
}}

Question:
"Which district has the highest total budget?"

Plan:
{{
  "operation": "group_sum",
  "group_by": "District",
  "field": "Cost (M)",
  "order": "desc"
}}

Question:
"How many projects are there by category?"

Plan:
{{
  "operation": "group",
  "group_by": "Category"
}}

Question:
"Show projects in Khuzdar with progress below 50%."

Plan:
{{
  "operation": "list",
  "limit": 10,
  "filters": [
    {{"field": "District", "operator": "eq", "value": "Khuzdar"}},
    {{"field": "Progress %", "operator": "lt", "value": 50}}
  ]
}}

Question:
"Show projects with no contractor."

Plan:
{{
  "operation": "list",
  "filters": [
    {{"field": "Contractor", "operator": "is_null"}}
  ]
}}

Question:
"How many projects have a budget greater than 50 million?"

Plan:
{{
  "operation": "count",
  "filters": [
    {{"field": "Cost (M)", "operator": "gt", "value": 50}}
  ]
}}

IMPORTANT:
- Return ONLY valid JSON.
- Never answer in natural language.
- Never invent column names.
- Use "Cost (M)" for budget/cost.
- Use "District" for districts.
- Use "Category" for categories.
- Use "Status" for project status.
- Use "Progress %" for progress.
- For "water", use Category = PHE.
- Filters are AND conditions.
- For top/most expensive use order desc.
- For cheapest/least expensive use order asc.
- Default list limit is 10.
- Maximum list/ranking limit is 100.

User question:
{question}
"""

    try:
        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        plan = _extract_json(
            response.content
        )

        if isinstance(plan, dict):
            return _normalize_plan(
                plan,
                fallback,
            )

    except Exception as exc:
        print(
            "[TRACK A] Groq planner error:",
            repr(exc),
        )

    return fallback


# ============================================================
# 8. PLAN NORMALIZATION
# ============================================================

def _normalize_plan(
    plan: dict,
    fallback: dict,
) -> dict:

    if not isinstance(plan, dict):

        return fallback

    allowed_operations = {
        "count",
        "sum",
        "average",
        "min",
        "max",
        "unique",
        "group",
        "group_sum",
        "group_average",
        "list",
        "top",
        "bottom",
    }

    operation = str(
        plan.get(
            "operation",
            fallback.get(
                "operation",
                "count",
            ),
        )
    ).lower().strip()

    if operation not in allowed_operations:

        operation = fallback.get(
            "operation",
            "count",
        )

    result = dict(plan)

    result["operation"] = operation

    if "limit" in result:

        try:

            result["limit"] = max(
                1,
                min(
                    int(result["limit"]),
                    100,
                ),
            )

        except Exception:

            result["limit"] = fallback.get(
                "limit",
                10,
            )

    else:

        result["limit"] = fallback.get(
            "limit",
            10,
        )

    if "filters" not in result:

        result["filters"] = fallback.get(
            "filters",
            [],
        )

    if not isinstance(
        result["filters"],
        list,
    ):

        result["filters"] = fallback.get(
            "filters",
            [],
        )

    if (
        operation in {
            "sum",
            "average",
            "min",
            "max",
            "top",
            "bottom",
            "group_sum",
            "group_average",
        }
        and not result.get("field")
    ):

        result["field"] = "Cost (M)"

    if operation in {
        "top",
        "bottom",
    }:

        if "order" not in result:

            result["order"] = (
                "asc"
                if operation == "bottom"
                else "desc"
            )

    return result


# ============================================================
# 9. EXECUTE PLAN
# ============================================================

def _execute_plan(
    plan: dict,
) -> dict:

    operation = plan.get(
        "operation",
        "count",
    )

    filters = plan.get(
        "filters",
        [],
    )

    field = plan.get(
        "field"
    )

    group_by = plan.get(
        "group_by"
    )

    limit = plan.get(
        "limit",
        10,
    )

    order = plan.get(
        "order",
        "desc",
    )

    # --------------------------------------------------------
    # Use the universal query engine
    # --------------------------------------------------------

    return query_projects(
        operation=operation,
        field=field,
        group_by=group_by,
        filters=filters,
        limit=limit,
        order=order,
    )


# ============================================================
# 10. RESULT FORMATTING
# ============================================================

def _format_number(
    value,
) -> str:

    if value is None:
        return "N/A"

    try:

        number = float(value)

        if number.is_integer():

            return f"{int(number):,}"

        return f"{number:,.2f}"

    except Exception:

        return str(value)


def _format_answer(
    question: str,
    plan: dict,
    result: dict,
) -> str:

    operation = plan.get(
        "operation",
        "count",
    )

    # ========================================================
    # COUNT
    # ========================================================

    if operation == "count":

        count = result.get(
            "count",
            0,
        )

        return (
            f"There are {_format_number(count)} "
            f"matching projects."
        )

    # ========================================================
    # SUM
    # ========================================================

    if operation == "sum":

        value = result.get(
            "value"
        )

        count = result.get(
            "count"
        )

        return (
            f"The total budget is "
            f"PKR {_format_number(value)} million "
            f"across {_format_number(count)} projects."
        )

    # ========================================================
    # AVERAGE
    # ========================================================

    if operation == "average":

        value = result.get(
            "value"
        )

        return (
            f"The average cost is "
            f"PKR {_format_number(value)} million."
        )

    # ========================================================
    # MIN / MAX
    # ========================================================

    if operation == "min":

        return (
            f"The minimum "
            f"{result.get('field', 'value')} "
            f"is {_format_number(result.get('value'))}."
        )

    if operation == "max":

        return (
            f"The maximum "
            f"{result.get('field', 'value')} "
            f"is {_format_number(result.get('value'))}."
        )

    # ========================================================
    # UNIQUE
    # ========================================================

    if operation == "unique":

        values = result.get(
            "values",
            [],
        )

        if not values:

            return "No values were found."

        return (
            f"There are {len(values)} unique "
            f"{result.get('field', 'values')}: "
            + ", ".join(
                str(value)
                for value in values
            )
        )

    # ========================================================
    # GROUP
    # ========================================================

    if operation == "group":

        rows = result.get(
            "results",
            [],
        )

        if not rows:

            return "No matching grouped data was found."

        lines = [
            f"- {row['group']}: "
            f"{_format_number(row['count'])} projects"
            for row in rows
        ]

        return (
            f"Projects grouped by "
            f"{result.get('group_by')}:\n"
            + "\n".join(lines)
        )

    # ========================================================
    # GROUP SUM / AVERAGE
    # ========================================================

    if operation in {
        "group_sum",
        "group_average",
    }:

        rows = result.get(
            "results",
            [],
        )

        if not rows:

            return "No grouped data was found."

        if operation == "group_sum":

            label = "total budget"

        else:

            label = "average cost"

        lines = []

        for row in rows:

            lines.append(
                f"- {row['group']}: "
                f"PKR {_format_number(row['value'])} million"
            )

        return (
            f"{label.title()} by "
            f"{result.get('group_by')}:\n"
            + "\n".join(lines)
        )

    # ========================================================
    # LIST
    # ========================================================

    if operation == "list":

        rows = result.get(
            "results",
            [],
        )

        if not rows:

            return "No matching projects were found."

        lines = []

        for index, row in enumerate(
            rows,
            start=1,
        ):

            project_number = row.get(
                "#",
                row.get(
                    "Global ID",
                    "N/A",
                ),
            )

            district = row.get(
                "District",
                "N/A",
            )

            category = row.get(
                "Category",
                "N/A",
            )

            description = row.get(
                "Description",
                "",
            )

            cost = row.get(
                "Cost (M)",
                "N/A",
            )

            status = row.get(
                "Status",
                "N/A",
            )

            lines.append(
                f"{index}. "
                f"Project {project_number} | "
                f"{district} | "
                f"{category} | "
                f"PKR {_format_number(cost)}M | "
                f"{status} | "
                f"{description}"
            )

        return (
            f"Found {result.get('count', len(rows))} "
            f"matching projects. Showing "
            f"{len(rows)}:\n"
            + "\n".join(lines)
        )

    # ========================================================
    # TOP / BOTTOM
    # ========================================================

    if operation in {
        "top",
        "bottom",
    }:

        rows = result.get(
            "results",
            [],
        )

        if not rows:

            return "No matching projects were found."

        if operation == "top":

            heading = "Most expensive"

        else:

            heading = "Least expensive"

        lines = []

        for index, row in enumerate(
            rows,
            start=1,
        ):

            number = row.get(
                "#",
                row.get(
                    "Global ID",
                    "N/A",
                ),
            )

            district = row.get(
                "District",
                "N/A",
            )

            category = row.get(
                "Category",
                "N/A",
            )

            cost = row.get(
                "Cost (M)",
                "N/A",
            )

            status = row.get(
                "Status",
                "N/A",
            )

            description = row.get(
                "Description",
                "",
            )

            lines.append(
                f"{index}. "
                f"Project {number} | "
                f"{district} | "
                f"{category} | "
                f"PKR {_format_number(cost)}M | "
                f"{status} | "
                f"{description}"
            )

        return (
            f"{heading} projects:\n"
            + "\n".join(lines)
        )

    return str(result)


# ============================================================
# 11. LANGGRAPH NODE
# ============================================================

def agent(state: MessagesState):

    if llm is None:

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "Groq is not configured. "
                        "Please configure GROQ_API_KEY."
                    ),
                }
            ]
        }

    response = llm.bind_tools(
        TOOLS
    ).invoke(
        state["messages"]
    )

    return {
        "messages": [
            response
        ]
    }


builder = StateGraph(
    MessagesState
)

builder.add_node(
    "agent",
    agent,
)

builder.add_node(
    "tools",
    ToolNode(TOOLS),
)

builder.add_edge(
    START,
    "agent",
)

builder.add_edge(
    "agent",
    "__end__",
)

graph = builder.compile()


# ============================================================
# 12. MAIN TRACK A API
# ============================================================

def run_track_a_traced(
    question: str,
) -> dict:

    question = str(
        question or ""
    ).strip()

    if not question:

        return {
            "question": "",
            "answer": "Please enter a question.",
            "trace": [],
            "steps": 0,
        }

    print(
        "\n[TRACK A] User question:",
        question,
    )

    # --------------------------------------------------------
    # Create query plan
    # --------------------------------------------------------

    plan = _create_plan(
        question
    )

    print(
        "[TRACK A] Query plan:",
        plan,
    )

    try:

        # ----------------------------------------------------
        # Execute against Pandas
        # ----------------------------------------------------

        result = _execute_plan(
            plan
        )

        print(
            "[TRACK A] Dataset result:",
            result,
        )

        # ----------------------------------------------------
        # Deterministic final answer
        # ----------------------------------------------------

        answer = _format_answer(
            question,
            plan,
            result,
        )

        trace = [
            {
                "step": 1,
                "agent": "Groq",
                "action": "plan",
                "result": plan,
            },
            {
                "step": 2,
                "tool": "query_projects",
                "arguments": plan,
                "result": result,
            },
        ]

        return {
            "question": question,
            "answer": answer,
            "trace": trace,
            "steps": len(trace),
        }

    except Exception as exc:

        print(
            "[TRACK A] Error:",
            repr(exc),
        )

        return {
            "question": question,
            "answer": (
                "I couldn't complete that data query. "
                f"Error: {exc}"
            ),
            "trace": [
                {
                    "step": 1,
                    "error": str(exc),
                    "plan": plan,
                }
            ],
            "steps": 1,
        }


def run_track_a(
    question: str,
) -> str:

    return run_track_a_traced(
        question
    )["answer"]


def run_track_a_detailed(
    question: str,
) -> dict:

    return run_track_a_traced(
        question
    )


# ============================================================
# 13. TERMINAL TEST MODE
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk the BSDI Agent: "
    ).strip()

    result = run_track_a_traced(
        question
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRACK A — BSDI DATA ASSISTANT"
    )

    print(
        "=" * 70
    )

    print(
        "\nANSWER:"
    )

    print(
        result["answer"]
    )

    print(
        "\nTRACE:"
    )

    print(
        json.dumps(
            result["trace"],
            indent=2,
            default=str,
        )
    )
