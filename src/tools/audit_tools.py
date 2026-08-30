from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.excel_loader import load_projects


# ==================================================
# DATASET
# ==================================================

DATA_PATH = Path("data/Projects.xlsx")

PROJECTS_DF = load_projects(DATA_PATH)


# ==================================================
# HELPERS
# ==================================================

def _clean_text(value):
    """
    Convert a value to a clean lowercase string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def _is_blank(value):
    """
    Check whether a dataset value is blank/missing.
    """
    if pd.isna(value):
        return True

    return str(value).strip() == ""


def _project_columns(df):
    """
    Return useful columns for audit results.
    """

    columns = [
        "#",
        "Global ID",
        "District",
        "Category",
        "Description",
        "Cost (M)",
        "NITs",
        "Contractor",
        "Status",
        "Work Started",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    return available


def _records(df):
    """
    Convert dataframe rows into JSON-friendly dictionaries.
    """

    result = df.copy()

    result = result.where(
        pd.notna(result),
        None,
    )

    return result.to_dict(
        orient="records"
    )


# ==================================================
# AUDIT 1
# ==================================================

def audit_missing_work_started():
    """
    Find projects that are marked In Progress
    but have no Work Started date.
    """

    df = PROJECTS_DF.copy()

    status_mask = (
        df["Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "in progress"
    )

    missing_date_mask = (
        df["Work Started"].isna()
        |
        (
            df["Work Started"]
            .astype(str)
            .str.strip()
            == ""
        )
    )

    findings = df[
        status_mask
        & missing_date_mask
    ]

    columns = _project_columns(findings)

    findings = findings[columns]

    return {
        "check": "in_progress_without_work_started",
        "description": (
            "Projects marked In Progress "
            "without a Work Started date."
        ),
        "count": int(len(findings)),
        "findings": _records(findings),
    }


# ==================================================
# AUDIT 2
# ==================================================

def audit_high_cost_no_contractor():
    """
    Find projects in the top 10% by cost
    that have no contractor assigned.
    """

    df = PROJECTS_DF.copy()

    df["Cost (M)"] = pd.to_numeric(
        df["Cost (M)"],
        errors="coerce",
    )

    valid_costs = df["Cost (M)"].dropna()

    if len(valid_costs) == 0:
        return {
            "check": "high_cost_without_contractor",
            "description": (
                "High-cost projects without "
                "an assigned contractor."
            ),
            "threshold_m_pkr": None,
            "count": 0,
            "findings": [],
        }

    threshold = valid_costs.quantile(0.90)

    high_cost = (
        df["Cost (M)"]
        >= threshold
    )

    no_contractor = (
        df["Contractor"].isna()
        |
        (
            df["Contractor"]
            .astype(str)
            .str.strip()
            == ""
        )
    )

    findings = df[
        high_cost
        & no_contractor
    ]

    columns = _project_columns(findings)

    findings = findings[columns]

    return {
        "check": "high_cost_without_contractor",
        "description": (
            "Projects in the top 10% by cost "
            "without an assigned contractor."
        ),
        "threshold_m_pkr": float(threshold),
        "count": int(len(findings)),
        "findings": _records(findings),
    }


# ==================================================
# AUDIT 3
# ==================================================

def audit_not_started_budget_by_district():
    """
    Identify districts where a large share of
    the district's budget is still Not Started.

    A district is flagged when >= 50% of its
    total project budget is Not Started.
    """

    df = PROJECTS_DF.copy()

    df["Cost (M)"] = pd.to_numeric(
        df["Cost (M)"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Cost (M)"]
    )

    if len(df) == 0:
        return {
            "check": "district_not_started_budget",
            "description": (
                "Districts with a large share "
                "of budget still Not Started."
            ),
            "count": 0,
            "findings": [],
        }

    total_budget = (
        df.groupby("District")["Cost (M)"]
        .sum()
    )

    not_started_df = df[
        df["Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "not started"
    ]

    not_started_budget = (
        not_started_df
        .groupby("District")["Cost (M)"]
        .sum()
    )

    results = []

    for district, budget in total_budget.items():

        ns_budget = float(
            not_started_budget.get(
                district,
                0.0,
            )
        )

        share = (
            ns_budget / budget
            if budget > 0
            else 0.0
        )

        if share >= 0.50:

            results.append(
                {
                    "district": str(district),
                    "total_budget_m_pkr": float(
                        budget
                    ),
                    "not_started_budget_m_pkr": ns_budget,
                    "not_started_budget_share": round(
                        share * 100,
                        2,
                    ),
                }
            )

    results.sort(
        key=lambda x: x[
            "not_started_budget_share"
        ],
        reverse=True,
    )

    return {
        "check": "district_not_started_budget",
        "description": (
            "Districts where at least 50% "
            "of the project budget is Not Started."
        ),
        "threshold_percent": 50,
        "count": len(results),
        "findings": results,
    }


# ==================================================
# AUDIT 4
# ==================================================

def audit_category_cost_outliers():
    """
    Find unusually expensive projects within
    their own category using the IQR method.

    A project is flagged when:

        Cost > Q3 + 1.5 * IQR
    """

    df = PROJECTS_DF.copy()

    df["Cost (M)"] = pd.to_numeric(
        df["Cost (M)"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Cost (M)", "Category"]
    )

    findings = []

    for category, group in df.groupby(
        "Category"
    ):

        if len(group) < 4:
            continue

        q1 = group["Cost (M)"].quantile(
            0.25
        )

        q3 = group["Cost (M)"].quantile(
            0.75
        )

        iqr = q3 - q1

        upper_limit = (
            q3 + (1.5 * iqr)
        )

        outliers = group[
            group["Cost (M)"]
            > upper_limit
        ].copy()

        for _, row in outliers.iterrows():

            findings.append(
                {
                    "project_number": (
                        row["#"]
                        if "#" in row
                        else None
                    ),
                    "global_id": (
                        row["Global ID"]
                        if "Global ID" in row
                        else None
                    ),
                    "district": row["District"],
                    "category": row["Category"],
                    "description": row[
                        "Description"
                    ],
                    "cost_m_pkr": float(
                        row["Cost (M)"]
                    ),
                    "category_q3_m_pkr": float(
                        q3
                    ),
                    "category_upper_limit_m_pkr": float(
                        upper_limit
                    ),
                    "status": row["Status"],
                }
            )

    findings.sort(
        key=lambda x: x[
            "cost_m_pkr"
        ],
        reverse=True,
    )

    return {
        "check": "category_cost_outliers",
        "description": (
            "Projects with unusually high costs "
            "compared with projects in the same category."
        ),
        "method": "IQR",
        "count": len(findings),
        "findings": findings,
    }


# ==================================================
# AUDIT 5
# ==================================================

def audit_nits_no_in_progress():
    """
    Find projects where:

        NITs = No
        AND
        Status = In Progress
    """

    df = PROJECTS_DF.copy()

    nits_mask = (
        df["NITs"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "no"
    )

    status_mask = (
        df["Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "in progress"
    )

    findings = df[
        nits_mask
        & status_mask
    ]

    columns = _project_columns(findings)

    findings = findings[columns]

    return {
        "check": "nits_no_but_in_progress",
        "description": (
            "Projects marked In Progress "
            "while NITs is No."
        ),
        "count": int(len(findings)),
        "findings": _records(findings),
    }


# ==================================================
# RUN ONE AUDIT CHECK
# ==================================================

def run_audit_check(
    check: str,
) -> dict:
    """
    Run one of the available audit checks.

    The agent can request a check by name.
    """

    check = (
        check
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    checks = {
        "in_progress_without_work_started":
            audit_missing_work_started,

        "missing_work_started":
            audit_missing_work_started,

        "high_cost_without_contractor":
            audit_high_cost_no_contractor,

        "district_not_started_budget":
            audit_not_started_budget_by_district,

        "category_cost_outliers":
            audit_category_cost_outliers,

        "nits_no_but_in_progress":
            audit_nits_no_in_progress,
    }

    if check not in checks:

        return {
            "error": (
                f"Unknown audit check: {check}"
            ),
            "available_checks": list(
                checks.keys()
            ),
        }

    return checks[check]()