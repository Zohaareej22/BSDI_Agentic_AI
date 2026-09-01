from pathlib import Path
from typing import Optional, Any

import pandas as pd

from src.ingestion.excel_loader import load_projects


# ============================================================
# DATASET
# ============================================================

DATA_PATH = Path("data/Projects.xlsx")

PROJECTS_DF = load_projects(DATA_PATH)


# ============================================================
# ALIASES
# ============================================================

CATEGORY_ALIASES = {
    "water": "PHE",
    "water projects": "PHE",
    "phe": "PHE",

    "education": "Education",
    "education projects": "Education",

    "health": "Health",
    "healthcare": "Health",
    "health projects": "Health",

    "road": "Road",
    "roads": "Road",
    "road projects": "Road",

    "irrigation": "Irrigation",

    "agriculture": "Agriculture",

    "energy": "Energy",

    "building": "Building",

    "municipal": "Municipal",

    "sewerage": "Sewerage",

    "security": "Security",

    "sports": "Sports",

    "other": "Other",
}


STATUS_ALIASES = {
    "completed": "Completed",
    "complete": "Completed",

    "in progress": "In Progress",
    "ongoing": "In Progress",
    "ongoing projects": "In Progress",

    "not started": "Not Started",
    "not-started": "Not Started",

    "nits issued": "NITs Issued",
    "nit issued": "NITs Issued",
}


COLUMN_ALIASES = {
    "cost": "Cost (M)",
    "budget": "Cost (M)",
    "total cost": "Cost (M)",
    "total budget": "Cost (M)",
    "project cost": "Cost (M)",

    "progress": "Progress %",
    "progress %": "Progress %",

    "district": "District",
    "category": "Category",
    "status": "Status",
    "phase": "Phase",

    "description": "Description",
    "project": "Description",
    "project name": "Description",

    "contractor": "Contractor",
    "agency": "Executing Agency",
    "executing agency": "Executing Agency",

    "nits": "NITs",

    "work started": "Work Started",

    "xen": "XEN Name",
    "xen name": "XEN Name",
    "xen contact": "XEN Contact",

    "global id": "Global ID",
    "id": "Global ID",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_category(category: Optional[str]) -> Optional[str]:

    if not category:
        return None

    cleaned = str(category).strip().lower()

    return CATEGORY_ALIASES.get(
        cleaned,
        str(category).strip(),
    )


def normalize_status(status: Optional[str]) -> Optional[str]:

    if not status:
        return None

    cleaned = str(status).strip().lower()

    return STATUS_ALIASES.get(
        cleaned,
        str(status).strip(),
    )


def resolve_column(column: Optional[str]) -> Optional[str]:

    if not column:
        return None

    cleaned = str(column).strip()

    if cleaned in PROJECTS_DF.columns:
        return cleaned

    lower = cleaned.lower()

    if lower in COLUMN_ALIASES:
        return COLUMN_ALIASES[lower]

    for actual in PROJECTS_DF.columns:

        if str(actual).strip().lower() == lower:
            return actual

    return cleaned


# ============================================================
# FILTER ENGINE
# ============================================================

def _apply_filter(
    df: pd.DataFrame,
    field: str,
    operator: str,
    value: Any,
) -> pd.DataFrame:

    field = resolve_column(field)

    if field not in df.columns:
        raise ValueError(
            f"Unknown field '{field}'. "
            f"Available fields: {list(df.columns)}"
        )

    operator = str(operator or "eq").lower().strip()

    series = df[field]

    # --------------------------------------------------------
    # Numeric comparisons
    # --------------------------------------------------------

    numeric_operators = {
        "gt",
        "gte",
        "lt",
        "lte",
        "eq",
        "neq",
    }

    if operator in numeric_operators:

        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):

            numeric_value = None

        if numeric_value is not None:

            if operator == "gt":
                return df[numeric_series > numeric_value]

            if operator == "gte":
                return df[numeric_series >= numeric_value]

            if operator == "lt":
                return df[numeric_series < numeric_value]

            if operator == "lte":
                return df[numeric_series <= numeric_value]

            if operator == "eq":
                return df[numeric_series == numeric_value]

            if operator == "neq":
                return df[numeric_series != numeric_value]

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    if operator in {
        "is_null",
        "missing",
        "blank",
    }:

        return df[
            series.isna()
            | series.astype(str).str.strip().eq("")
        ]

    if operator in {
        "not_null",
        "not_missing",
        "not_blank",
    }:

        return df[
            series.notna()
            & ~series.astype(str).str.strip().eq("")
        ]

    # --------------------------------------------------------
    # Text comparisons
    # --------------------------------------------------------

    text = series.astype(str).str.strip().str.lower()

    target = str(value).strip().lower()

    if operator == "eq":

        return df[text == target]

    if operator == "neq":

        return df[text != target]

    if operator in {
        "contains",
        "like",
    }:

        return df[
            text.str.contains(
                target,
                case=False,
                na=False,
            )
        ]

    if operator == "starts_with":

        return df[
            text.str.startswith(
                target,
                na=False,
            )
        ]

    if operator == "ends_with":

        return df[
            text.str.endswith(
                target,
                na=False,
            )
        ]

    raise ValueError(
        f"Unsupported filter operator: {operator}"
    )


def apply_filters(
    df: pd.DataFrame,
    filters: Optional[list[dict]] = None,
) -> pd.DataFrame:

    if not filters:
        return df

    result = df.copy()

    for condition in filters:

        if not isinstance(condition, dict):
            continue

        field = condition.get("field")
        operator = condition.get(
            "operator",
            "eq",
        )
        value = condition.get("value")

        if not field:
            continue

        result = _apply_filter(
            result,
            field,
            operator,
            value,
        )

    return result


# ============================================================
# SIMPLE FILTERS
# ============================================================

def _apply_simple_filters(
    df: pd.DataFrame,
    district: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> pd.DataFrame:

    result = df.copy()

    if district:

        result = _apply_filter(
            result,
            "District",
            "eq",
            district,
        )

    if category:

        category = normalize_category(category)

        result = _apply_filter(
            result,
            "Category",
            "eq",
            category,
        )

    if status:

        status = normalize_status(status)

        result = _apply_filter(
            result,
            "Status",
            "eq",
            status,
        )

    return result


# ============================================================
# UNIVERSAL QUERY
# ============================================================

def query_projects(
    district: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    operation: str = "count",
    field: Optional[str] = None,
    group_by: Optional[str] = None,
    filters: Optional[list[dict]] = None,
    limit: int = 10,
    order: str = "desc",
) -> dict:

    df = PROJECTS_DF.copy()

    # Simple filters
    df = _apply_simple_filters(
        df,
        district=district,
        category=category,
        status=status,
    )

    # Advanced filters
    df = apply_filters(
        df,
        filters,
    )

    operation = str(
        operation or "count"
    ).strip().lower()

    order = str(
        order or "desc"
    ).strip().lower()

    if order not in {
        "asc",
        "desc",
    }:
        order = "desc"

    limit = max(
        1,
        min(
            int(limit or 10),
            100,
        ),
    )

    # ========================================================
    # COUNT
    # ========================================================

    if operation == "count":

        return {
            "operation": "count",
            "count": int(len(df)),
        }

    # ========================================================
    # SUM
    # ========================================================

    if operation in {
        "sum",
        "total",
        "total_cost",
        "total_budget",
    }:

        field = resolve_column(
            field or "Cost (M)"
        )

        values = pd.to_numeric(
            df[field],
            errors="coerce",
        )

        return {
            "operation": "sum",
            "field": field,
            "value": float(
                values.sum()
            ),
            "count": int(
                values.notna().sum()
            ),
        }

    # ========================================================
    # AVERAGE
    # ========================================================

    if operation in {
        "average",
        "avg",
        "mean",
        "average_cost",
    }:

        field = resolve_column(
            field or "Cost (M)"
        )

        values = pd.to_numeric(
            df[field],
            errors="coerce",
        )

        return {
            "operation": "average",
            "field": field,
            "value": (
                float(values.mean())
                if values.notna().any()
                else None
            ),
            "count": int(
                values.notna().sum()
            ),
        }

    # ========================================================
    # MINIMUM
    # ========================================================

    if operation in {
        "min",
        "minimum",
        "lowest",
    }:

        field = resolve_column(
            field or "Cost (M)"
        )

        values = pd.to_numeric(
            df[field],
            errors="coerce",
        )

        return {
            "operation": "min",
            "field": field,
            "value": (
                float(values.min())
                if values.notna().any()
                else None
            ),
        }

    # ========================================================
    # MAXIMUM
    # ========================================================

    if operation in {
        "max",
        "maximum",
        "highest",
        "largest",
    }:

        field = resolve_column(
            field or "Cost (M)"
        )

        values = pd.to_numeric(
            df[field],
            errors="coerce",
        )

        return {
            "operation": "max",
            "field": field,
            "value": (
                float(values.max())
                if values.notna().any()
                else None
            ),
        }

    # ========================================================
    # UNIQUE VALUES
    # ========================================================

    if operation in {
        "unique",
        "distinct",
        "values",
    }:

        field = resolve_column(field)

        if not field:
            raise ValueError(
                "A field is required for unique values."
            )

        values = (
            df[field]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )

        return {
            "operation": "unique",
            "field": field,
            "count": len(values),
            "values": values[:100],
        }

    # ========================================================
    # GROUP
    # ========================================================

    if operation in {
        "group",
        "group_by",
        "group_count",
        "distribution",
    }:

        group_field = resolve_column(
            group_by or field
        )

        if not group_field:
            raise ValueError(
                "group_by is required for grouping."
            )

        grouped = (
            df[group_field]
            .fillna("Missing")
            .astype(str)
            .str.strip()
            .value_counts()
        )

        results = [
            {
                "group": str(index),
                "count": int(value),
            }
            for index, value in grouped.items()
        ]

        return {
            "operation": "group",
            "group_by": group_field,
            "results": results,
        }

    # ========================================================
    # GROUP + SUM
    # ========================================================

    if operation in {
        "group_sum",
        "sum_by",
        "total_by",
    }:

        group_field = resolve_column(
            group_by
        )

        value_field = resolve_column(
            field or "Cost (M)"
        )

        if not group_field:
            raise ValueError(
                "group_by is required."
            )

        numeric_values = pd.to_numeric(
            df[value_field],
            errors="coerce",
        )

        working = df.copy()

        working["_numeric_value"] = numeric_values

        grouped = (
            working
            .groupby(group_field, dropna=False)
            ["_numeric_value"]
            .sum()
            .sort_values(
                ascending=(
                    order == "asc"
                )
            )
        )

        results = [
            {
                "group": (
                    "Missing"
                    if pd.isna(index)
                    else str(index)
                ),
                "value": float(value),
            }
            for index, value in grouped.items()
        ]

        return {
            "operation": "group_sum",
            "group_by": group_field,
            "field": value_field,
            "results": results,
        }

    # ========================================================
    # GROUP + AVERAGE
    # ========================================================

    if operation in {
        "group_average",
        "average_by",
        "avg_by",
    }:

        group_field = resolve_column(
            group_by
        )

        value_field = resolve_column(
            field or "Cost (M)"
        )

        if not group_field:
            raise ValueError(
                "group_by is required."
            )

        working = df.copy()

        working["_numeric_value"] = pd.to_numeric(
            working[value_field],
            errors="coerce",
        )

        grouped = (
            working
            .groupby(group_field, dropna=False)
            ["_numeric_value"]
            .mean()
            .sort_values(
                ascending=(
                    order == "asc"
                )
            )
        )

        results = [
            {
                "group": (
                    "Missing"
                    if pd.isna(index)
                    else str(index)
                ),
                "value": (
                    float(value)
                    if pd.notna(value)
                    else None
                ),
            }
            for index, value in grouped.items()
        ]

        return {
            "operation": "group_average",
            "group_by": group_field,
            "field": value_field,
            "results": results,
        }

    # ========================================================
    # LIST
    # ========================================================

    if operation in {
        "list",
        "show",
        "filter",
    }:

        clean = df.copy()

        clean = clean.where(
            pd.notna(clean),
            None,
        )

        records = clean.to_dict(
            orient="records"
        )

        return {
            "operation": "list",
            "count": len(records),
            "results": records[:limit],
        }

    # ========================================================
    # TOP / BOTTOM
    # ========================================================

    if operation in {
        "top",
        "bottom",
        "rank",
    }:

        value_field = resolve_column(
            field or "Cost (M)"
        )

        working = df.copy()

        working["_numeric_value"] = pd.to_numeric(
            working[value_field],
            errors="coerce",
        )

        ascending = (
            operation == "bottom"
            or order == "asc"
        )

        working = working.sort_values(
            "_numeric_value",
            ascending=ascending,
        )

        top = working.head(limit)

        results = []

        for _, row in top.iterrows():

            record = {}

            for column in df.columns:

                value = row[column]

                if pd.isna(value):
                    value = None

                elif hasattr(
                    value,
                    "item",
                ):

                    try:
                        value = value.item()
                    except Exception:
                        pass

                record[column] = value

            results.append(record)

        return {
            "operation": operation,
            "field": value_field,
            "limit": limit,
            "results": results,
        }

    raise ValueError(
        f"Unsupported operation '{operation}'."
    )


# ============================================================
# GROUP PROJECTS
# ============================================================

def group_projects(
    group_by: str,
    category: Optional[str] = None,
    status: Optional[str] = None,
    district: Optional[str] = None,
) -> dict:

    return query_projects(
        district=district,
        category=category,
        status=status,
        operation="group",
        group_by=group_by,
    )


# ============================================================
# FILTER PROJECTS
# ============================================================

def filter_projects(
    district: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    filters: Optional[list[dict]] = None,
    limit: int = 100,
) -> list[dict]:

    result = query_projects(
        district=district,
        category=category,
        status=status,
        filters=filters,
        operation="list",
        limit=limit,
    )

    return result.get(
        "results",
        [],
    )


# ============================================================
# RANK PROJECTS
# ============================================================

def rank_projects(
    category: Optional[str] = None,
    status: Optional[str] = None,
    district: Optional[str] = None,
    limit: int = 5,
    order: str = "desc",
) -> dict:

    return query_projects(
        district=district,
        category=category,
        status=status,
        operation="top",
        field="Cost (M)",
        limit=limit,
        order=order,
    )
