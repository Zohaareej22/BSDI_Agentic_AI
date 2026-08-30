import os
import re
from typing import Any

from langchain_ollama import ChatOllama

from src.tools.project_tools import (
    query_projects,
    group_projects,
    rank_projects,
    filter_projects,
)


# ============================================================
# 1. OLLAMA CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://ollama:11434",
)

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "qwen3:4b",
)


# ============================================================
# 2. QWEN MODEL
# ============================================================

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    reasoning=False,
    disable_streaming=True,
    num_predict=250,
)


# ============================================================
# 3. CATEGORY EXTRACTION
# ============================================================

CATEGORY_ALIASES = {
    "water": "PHE",
    "water projects": "PHE",
    "phe": "PHE",

    "health": "Health",
    "health projects": "Health",

    "education": "Education",
    "education projects": "Education",

    "road": "Road",
    "roads": "Road",
    "road projects": "Road",

    "irrigation": "Irrigation",
    "irrigation projects": "Irrigation",
}


def _extract_category(question: str):
    q = question.lower()

    # Check longer phrases first
    phrases = sorted(
        CATEGORY_ALIASES.keys(),
        key=len,
        reverse=True,
    )

    for phrase in phrases:
        if phrase in q:
            return CATEGORY_ALIASES[phrase]

    return None


# ============================================================
# 4. STATUS EXTRACTION
# ============================================================

STATUS_ALIASES = {
    "completed": "Completed",
    "complete": "Completed",

    "in progress": "In Progress",
    "ongoing": "In Progress",

    "not started": "Not Started",
    "not-started": "Not Started",

    "nits issued": "NITs Issued",
}


def _extract_status(question: str):
    q = question.lower()

    phrases = sorted(
        STATUS_ALIASES.keys(),
        key=len,
        reverse=True,
    )

    for phrase in phrases:
        if phrase in q:
            return STATUS_ALIASES[phrase]

    return None


# ============================================================
# 5. DISTRICT EXTRACTION
# ============================================================

KNOWN_DISTRICTS = [
    "Kech",
    "Killa Abdullah",
    "Qilla Saifullah",
    "Lasbela",
    "Hub",
    "Loralai",
    "Chaghi",
    "Kohlu",
    "Naseerabad",
    "Zhob",
    "Gawadar",
    "Musa Khel",
    "Usta Muhammad",
    "Jhal Magsi",
    "Barkhan",
    "Nushki",
    "Surab",
    "Ziarat",
    "Kalat",
    "Washuk",
    "Tump",
    "Dera Bugti",
    "Jaffarabad",
    "Pishin",
    "Duki",
    "Harnai",
    "Barshore",
    "Upper Dera Bugti",
    "Kharan",
    "Mastung",
]


def _extract_district(question: str):
    q = question.lower()

    # Longest first prevents partial matches
    districts = sorted(
        KNOWN_DISTRICTS,
        key=len,
        reverse=True,
    )

    for district in districts:
        if district.lower() in q:
            return district

    return None


# ============================================================
# 6. LIMIT EXTRACTION
# ============================================================

def _extract_limit(question: str) -> int:
    match = re.search(
        r"\b(?:top|first|last|list|show)\s+(\d+)\b",
        question.lower(),
    )

    if match:
        return max(1, min(int(match.group(1)), 100))

    match = re.search(
        r"\b(\d+)\s+(?:most|least|highest|lowest)\b",
        question.lower(),
    )

    if match:
        return max(1, min(int(match.group(1)), 100))

    return 5


# ============================================================
# 7. TOOL SELECTION
# ============================================================

def _select_tool(question: str) -> str:

    q = question.lower().strip()

    # Ranking / cost questions
    if any(
        phrase in q
        for phrase in (
            "most expensive",
            "least expensive",
            "most costly",
            "least costly",
            "highest cost",
            "lowest cost",
            "highest-cost",
            "lowest-cost",
            "top projects",
            "most expensive projects",
        )
    ):
        return "rank_projects"

    # Grouping questions
    if (
        "which district" in q
        or "district has the most" in q
        or "district has the least" in q
        or "projects by district" in q
        or "by district" in q
        or "by category" in q
        or "by status" in q
    ):
        return "group_projects"

    # Explicit filtering/listing
    if any(
        phrase in q
        for phrase in (
            "list",
            "show",
            "find",
            "filter",
        )
    ):
        return "filter_projects"

    # Default
    return "query_projects"


# ============================================================
# 8. SAFE TOOL CALL
# ============================================================

def _call_tool(
    function,
    **kwargs,
):
    """
    Only pass parameters supported by the actual function.
    """

    import inspect

    signature = inspect.signature(function)

    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    return function(
        **valid_kwargs
    )


# ============================================================
# 9. EXECUTE TOOL
# ============================================================

def _execute_tool(
    tool_name: str,
    question: str,
):

    category = _extract_category(question)
    status = _extract_status(question)
    district = _extract_district(question)
    limit = _extract_limit(question)

    q = question.lower()

    print(
        f"[AGENT] Category: {category}"
    )

    print(
        f"[AGENT] Status: {status}"
    )

    print(
        f"[AGENT] District: {district}"
    )

    print(
        f"[AGENT] Limit: {limit}"
    )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    if tool_name == "rank_projects":

        descending = any(
            phrase in q
            for phrase in (
                "most expensive",
                "most costly",
                "highest cost",
                "highest-cost",
                "highest",
                "largest",
                "most",
            )
        )

        order = (
            "desc"
            if descending
            else "asc"
        )

        return _call_tool(
            rank_projects,
            category=category,
            status=status,
            district=district,
            limit=limit,
            order=order,
        )

    # --------------------------------------------------------
    # GROUP
    # --------------------------------------------------------

    if tool_name == "group_projects":

        if "district" in q:
            group_by = "district"

        elif "status" in q:
            group_by = "status"

        else:
            group_by = "category"

        return _call_tool(
            group_projects,
            group_by=group_by,
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    if tool_name == "filter_projects":

        return _call_tool(
            filter_projects,
            district=district,
            category=category,
            status=status,
        )

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    if tool_name == "query_projects":

        operation = "count"

        if any(
            phrase in q
            for phrase in (
                "total cost",
                "total budget",
                "overall cost",
                "total amount",
            )
        ):
            operation = "total_cost"

        elif any(
            phrase in q
            for phrase in (
                "average cost",
                "avg cost",
                "mean cost",
            )
        ):
            operation = "average_cost"

        # Your query_projects implementation may
        # support different signatures across revisions.
        try:
            return _call_tool(
                query_projects,
                operation=operation,
                category=category,
                status=status,
                district=district,
                limit=limit,
                question=question,
            )

        except TypeError:
            return _call_tool(
                query_projects,
                operation=operation,
            )

    raise ValueError(
        f"Unknown BSDI tool: {tool_name}"
    )


# ============================================================
# 10. RESULT HELPERS
# ============================================================

def _get_results(tool_result):

    if isinstance(
        tool_result,
        dict,
    ):
        results = tool_result.get(
            "results"
        )

        if isinstance(
            results,
            list,
        ):
            return results

    if isinstance(
        tool_result,
        list,
    ):
        return tool_result

    return []


# ============================================================
# 11. DETERMINISTIC ANSWER GENERATORS
# ============================================================

def _format_count_answer(
    question: str,
    tool_result: dict,
):

    count = tool_result.get(
        "count",
        0,
    )

    category = _extract_category(
        question
    )

    status = _extract_status(
        question
    )

    district = _extract_district(
        question
    )

    # Human-friendly category
    category_name = category

    if category == "PHE":
        category_name = "water"

    # Build natural answer
    parts = []

    if category_name:
        parts.append(
            f"{category_name} projects"
        )
    else:
        parts.append(
            "projects"
        )

    if district:
        parts.append(
            f"in {district}"
        )

    if status:
        parts.append(
            f"with status '{status}'"
        )

    description = " ".join(parts)

    return (
        f"There {'is' if count == 1 else 'are'} "
        f"{count} {description}."
    )


def _format_rank_answer(
    tool_result,
):

    results = _get_results(
        tool_result
    )

    if not results:
        return (
            "No matching projects were found."
        )

    lines = []

    for index, project in enumerate(
        results,
        start=1,
    ):

        if not isinstance(
            project,
            dict,
        ):
            lines.append(
                f"{index}. {project}"
            )
            continue

        number = project.get(
            "project_number",
            "N/A",
        )

        district = project.get(
            "district",
            "N/A",
        )

        category = project.get(
            "category",
            "N/A",
        )

        cost = project.get(
            "cost_m_pkr",
            "N/A",
        )

        status = project.get(
            "status",
            "N/A",
        )

        description = project.get(
            "description",
            "",
        )

        lines.append(
            f"{index}. Project {number} — "
            f"{district} — "
            f"{category} — "
            f"PKR {cost} million — "
            f"{status}"
        )

        if description:
            lines.append(
                f"   {description}"
            )

    return "\n".join(lines)


def _format_group_answer(
    question: str,
    tool_result: dict,
):

    results = _get_results(
        tool_result
    )

    if not results:
        return (
            "No grouping results were found."
        )

    q = question.lower()

    # Most
    if any(
        phrase in q
        for phrase in (
            "most",
            "highest",
            "largest",
        )
    ):

        best = max(
            results,
            key=lambda x: x.get(
                "count",
                0,
            ),
        )

        return (
            f"{best.get('group')} has the most "
            f"projects with "
            f"{best.get('count')} projects."
        )

    # Least
    if any(
        phrase in q
        for phrase in (
            "least",
            "fewest",
            "lowest",
        )
    ):

        best = min(
            results,
            key=lambda x: x.get(
                "count",
                0,
            ),
        )

        return (
            f"{best.get('group')} has the fewest "
            f"projects with "
            f"{best.get('count')} projects."
        )

    # Full distribution
    lines = []

    for item in results:
        lines.append(
            f"{item.get('group')}: "
            f"{item.get('count')} projects"
        )

    return "\n".join(lines)


def _format_filter_answer(
    tool_result,
):

    results = _get_results(
        tool_result
    )

    if not results:
        return (
            "No matching projects were found."
        )

    lines = []

    for index, project in enumerate(
        results,
        start=1,
    ):

        if not isinstance(
            project,
            dict,
        ):
            continue

        number = project.get(
            "Project Number",
            project.get(
                "project_number",
                "N/A",
            ),
        )

        district = project.get(
            "District",
            project.get(
                "district",
                "N/A",
            ),
        )

        category = project.get(
            "Category",
            project.get(
                "category",
                "N/A",
            ),
        )

        status = project.get(
            "Status",
            project.get(
                "status",
                "N/A",
            ),
        )

        cost = project.get(
            "Cost (M)",
            project.get(
                "cost_m_pkr",
                "N/A",
            ),
        )

        description = project.get(
            "Description",
            project.get(
                "description",
                "",
            ),
        )

        lines.append(
            f"{index}. Project {number} — "
            f"{district} — "
            f"{category} — "
            f"PKR {cost} million — "
            f"{status}"
        )

        if description:
            lines.append(
                f"   {description}"
            )

    return "\n".join(lines)


# ============================================================
# 12. FINAL ANSWER
# ============================================================

def _generate_final_answer(
    question: str,
    tool_name: str,
    tool_result: Any,
):

    # --------------------------------------------------------
    # COUNT / QUERY
    # --------------------------------------------------------

    if (
        tool_name == "query_projects"
        and isinstance(
            tool_result,
            dict,
        )
        and "count" in tool_result
    ):

        return _format_count_answer(
            question,
            tool_result,
        )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    if tool_name == "rank_projects":

        return _format_rank_answer(
            tool_result
        )

    # --------------------------------------------------------
    # GROUP
    # --------------------------------------------------------

    if tool_name == "group_projects":

        return _format_group_answer(
            question,
            tool_result,
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    if tool_name == "filter_projects":

        return _format_filter_answer(
            tool_result
        )

    # --------------------------------------------------------
    # FALLBACK — QWEN ONLY WHEN NECESSARY
    # --------------------------------------------------------

    prompt = f"""
You are the BSDI infrastructure data assistant.

Answer the user's question using ONLY the dataset result.

User question:
{question}

Dataset result:
{tool_result}

Rules:
- Give only the direct answer.
- Never invent facts.
- Never invent numbers.
- Never invent project information.
- Do not explain your reasoning.
- Do not mention tools.
- Do not say "let me analyze".
- Do not repeat the question.
- Keep the answer concise.
"""

    response = llm.invoke(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    content = str(
        response.content
    ).strip()

    # Remove accidental reasoning tags
    content = re.sub(
        r"<think>.*?</think>",
        "",
        content,
        flags=re.DOTALL,
    ).strip()

    return content


# ============================================================
# 13. MAIN TRACK A
# ============================================================

def run_track_a_traced(
    question: str,
) -> dict:

    question = (
        question or ""
    ).strip()

    if not question:
        return {
            "question": "",
            "answer": (
                "Please enter a question."
            ),
            "trace": [],
            "steps": 0,
        }

    tool_name = _select_tool(
        question
    )

    print(
        "\n[AGENT] Calling Qwen..."
    )

    print(
        "[AGENT] Selected:",
        tool_name,
    )

    try:

        tool_result = _execute_tool(
            tool_name,
            question,
        )

        print(
            "[AGENT] Dataset operation completed."
        )

        answer = _generate_final_answer(
            question,
            tool_name,
            tool_result,
        )

        print(
            "[AGENT] Final response generated."
        )

        trace = [
            {
                "tool": tool_name,
                "arguments": {
                    "question": question,
                    "category": _extract_category(
                        question
                    ),
                    "status": _extract_status(
                        question
                    ),
                    "district": _extract_district(
                        question
                    ),
                    "limit": _extract_limit(
                        question
                    ),
                },
                "result": tool_result,
            }
        ]

        return {
            "question": question,
            "answer": answer,
            "trace": trace,
            "steps": len(trace),
        }

    except Exception as exc:

        print(
            "[AGENT] Tool error:",
            repr(exc),
        )

        return {
            "question": question,
            "answer": (
                "I couldn't complete that "
                "BSDI data query. "
                f"Error: {exc}"
            ),
            "trace": [
                {
                    "tool": tool_name,
                    "arguments": {
                        "question": question,
                    },
                    "error": str(exc),
                }
            ],
            "steps": 1,
        }


# ============================================================
# 14. COMPATIBILITY FUNCTIONS
# ============================================================

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
# 15. TERMINAL MODE
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk the BSDI Agent: "
    ).strip()

    print(
        "\n" + "=" * 60
    )

    print(
        "TRACK A — BSDI DATA AGENT"
    )

    print(
        "=" * 60
    )

    result = run_track_a_traced(
        question
    )

    print(
        "\nFINAL AGENT RESPONSE"
    )

    print(
        "-" * 60
    )

    print(
        result["answer"]
    )

    print(
        "\nTRACE"
    )

    print(
        "-" * 60
    )

    for step in result.get(
        "trace",
        [],
    ):

        print(
            f"Tool: {step.get('tool')}"
        )

        print(
            f"Arguments: "
            f"{step.get('arguments')}"
        )

        print(
            f"Result: "
            f"{step.get('result')}"
        )
