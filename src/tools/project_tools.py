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
# NATURAL-LANGUAGE ALIASES
# ==================================================

CATEGORY_ALIASES = {
    "water": "PHE",
    "water projects": "PHE",
    "phe": "PHE",

    "education": "Education",
    "education projects": "Education",

    "health": "Health",
    "health projects": "Health",

    "road": "Road",
    "roads": "Road",
    "road projects": "Road",

    "irrigation": "Irrigation",
}


STATUS_ALIASES = {
    "completed": "Completed",
    "complete": "Completed",

    "in progress": "In Progress",
    "ongoing": "In Progress",

    "not started": "Not Started",
    "not-started": "Not Started",

    "nits issued": "NITs Issued",
}


# ==================================================
# NORMALIZATION
# ==================================================

def normalize_category(
    category: Optional[str],
) -> Optional[str]:

    if not category:
        return None

    cleaned = category.strip().lower()

    return CATEGORY_ALIASES.get(
        cleaned,
        category.strip(),
    )


def normalize_status(
    status: Optional[str],
) -> Optional[str]:

    if not status:
        return None

    cleaned = status.strip().lower()

    return STATUS_ALIASES.get(
        cleaned,
        status.strip(),
    )


# ==================================================
# QUERY PROJECTS
# ==================================================

def query_projects(
    district: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    operation: str = "count",
) -> dict:
    """
    Query the BSDI infrastructure project dataset.

    Supported filters:
        district
        category
        status

    Supported operations:
        count
        total_cost
        average_cost
    """

    df = PROJECTS_DF.copy()

    category = normalize_category(category)
    status = normalize_status(status)

    # District filter
    if district:
        district_clean = district.strip().lower()

        df = df[
            df["District"]
            .astype(str)
            .str.strip()
            .str.lower()
            == district_clean
        ]

    # Category filter
    if category:
        df = df[
            df["Category"]
            .astype(str)
            .str.strip()
            .str.lower()
            == category.lower()
        ]

    # Status filter
    if status:
        df = df[
            df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == status.lower()
        ]

    # Aggregation
    operation = operation.strip().lower()

    if operation == "count":
        return {
            "operation": "count",
            "count": int(len(df)),
        }

    if operation == "total_cost":

        costs = pd.to_numeric(
            df["Cost (M)"],
            errors="coerce",
        ).dropna()

        return {
            "operation": "total_cost",
            "total_cost_m_pkr": float(costs.sum()),
            "count": int(len(df)),
        }

    if operation == "average_cost":

        costs = pd.to_numeric(
            df["Cost (M)"],
            errors="coerce",
        ).dropna()

        average = (
            float(costs.mean())
            if len(costs) > 0
            else 0.0
        )

        return {
            "operation": "average_cost",
            "average_cost_m_pkr": average,
            "count": int(len(df)),
        }

    raise ValueError(
        f"Unsupported operation: {operation}. "
        "Use count, total_cost, or average_cost."
    )


# ==================================================
# GROUP PROJECTS
# ==================================================

def group_projects(
    group_by: str,
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    """
    Group projects and count them.

    Supported group_by:
        district
        category
        status
    """

    df = PROJECTS_DF.copy()

    category = normalize_category(category)
    status = normalize_status(status)

    # Category filter
    if category:
        df = df[
            df["Category"]
            .astype(str)
            .str.strip()
            .str.lower()
            == category.lower()
        ]

    # Status filter
    if status:
        df = df[
            df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == status.lower()
        ]

    # Allowed grouping columns
    column_map = {
        "district": "District",
        "category": "Category",
        "status": "Status",
    }

    group_by_clean = group_by.strip().lower()

    if group_by_clean not in column_map:
        raise ValueError(
            "group_by must be district, category, or status."
        )

    column = column_map[group_by_clean]

    # Group and count
    grouped = (
        df.groupby(column)
        .size()
        .sort_values(ascending=False)
    )

    return {
        "group_by": group_by_clean,
        "results": [
            {
                "group": str(index),
                "count": int(count),
            }
            for index, count in grouped.items()
        ],
    }


# ==================================================
# FILTER PROJECTS
# ==================================================

def filter_projects(
    district: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict]:
    """
    Return project rows matching the supplied filters.
    """

    df = PROJECTS_DF.copy()

    category = normalize_category(category)
    status = normalize_status(status)

    # District filter
    if district:
        district_clean = district.strip().lower()

        df = df[
            df["District"]
            .astype(str)
            .str.strip()
            .str.lower()
            == district_clean
        ]

    # Category filter
    if category:
        df = df[
            df["Category"]
            .astype(str)
            .str.strip()
            .str.lower()
            == category.lower()
        ]

    # Status filter
    if status:
        df = df[
            df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == status.lower()
        ]

    # Convert NaN → None
    df = df.where(
        pd.notna(df),
        None,
    )

    return df.to_dict(
        orient="records"
    )
# ==================================================
# RANK PROJECTS
# ==================================================

def rank_projects(
    category: Optional[str] = None,
    status: Optional[str] = None,
    district: Optional[str] = None,
    limit: int = 5,
    order: str = "desc",
) -> dict:
    """
    Rank projects by cost.

    Supported filters:
        category
        status
        district

    Parameters:
        limit: number of projects to return
        order: desc for most expensive,
               asc for least expensive
    """

    df = PROJECTS_DF.copy()

    category = normalize_category(category)
    status = normalize_status(status)

    # --------------------------------------------------
    # District filter
    # --------------------------------------------------

    if district:
        district_clean = district.strip().lower()

        df = df[
            df["District"]
            .astype(str)
            .str.strip()
            .str.lower()
            == district_clean
        ]

    # --------------------------------------------------
    # Category filter
    # --------------------------------------------------

    if category:
        df = df[
            df["Category"]
            .astype(str)
            .str.strip()
            .str.lower()
            == category.lower()
        ]

    # --------------------------------------------------
    # Status filter
    # --------------------------------------------------

    if status:
        df = df[
            df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == status.lower()
        ]

    # --------------------------------------------------
    # Convert cost to numeric
    # --------------------------------------------------

    df["Cost_numeric"] = pd.to_numeric(
        df["Cost (M)"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Cost_numeric"]
    )

    # --------------------------------------------------
    # Validate order
    # --------------------------------------------------

    order = order.strip().lower()

    if order not in {"asc", "desc"}:
        raise ValueError(
            "order must be 'asc' or 'desc'."
        )

    ascending = order == "asc"

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    df = df.sort_values(
        "Cost_numeric",
        ascending=ascending,
    )

    # --------------------------------------------------
    # Limit results
    # --------------------------------------------------

    limit = max(
        1,
        min(int(limit), 100)
    )

    top = df.head(limit)

    # --------------------------------------------------
    # Return clean results
    # --------------------------------------------------

    results = []

    for _, row in top.iterrows():

        results.append(
            {
                "project_number": row.get("#"),
                "global_id": row.get("Global ID"),
                "district": row.get("District"),
                "category": row.get("Category"),
                "description": row.get("Description"),
                "cost_m_pkr": float(
                    row["Cost_numeric"]
                ),
                "status": row.get("Status"),
            }
        )

    return {
        "order": order,
        "limit": limit,
        "results": results,
    }