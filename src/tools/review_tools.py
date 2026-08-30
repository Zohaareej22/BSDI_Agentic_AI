import pandas as pd
from pathlib import Path

from src.ingestion.excel_loader import load_projects


# ==================================================
# LOAD DATA
# ==================================================

DATA_PATH = Path("data/Projects.xlsx")

df = load_projects(DATA_PATH)


# ==================================================
# COMMON HELPER
# ==================================================

def get_not_started():
    """
    Return all currently Not Started projects.
    """

    return df[
        df["Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "not started"
    ].copy()


# ==================================================
# FINANCE AGENT ANALYSIS
# ==================================================

def finance_review(limit=10):
    """
    Finance Agent:
    Analyze Not Started projects from a
    financial perspective.
    """

    projects = get_not_started()

    if projects.empty:
        return {
            "agent": "Finance Agent",
            "projects": [],
            "total_not_started_budget_m_pkr": 0,
        }

    # Convert cost safely
    projects["Cost (M)"] = pd.to_numeric(
        projects["Cost (M)"],
        errors="coerce"
    ).fillna(0)

    total_budget = projects[
        "Cost (M)"
    ].sum()

    # Category average
    category_average = (
        projects
        .groupby("Category")["Cost (M)"]
        .mean()
        .to_dict()
    )

    # Rank financially attractive projects
    projects["category_avg"] = (
        projects["Category"]
        .map(category_average)
    )

    projects["cost_ratio"] = (
        projects["Cost (M)"]
        / projects["category_avg"].replace(
            0,
            pd.NA
        )
    )

    # Lower cost is generally easier
    # to fit into the PKR 2B budget.
    ranked = projects.sort_values(
        by="Cost (M)",
        ascending=True
    )

    results = []

    for _, row in ranked.head(limit).iterrows():

        results.append({
            "global_id": row.get(
                "Global ID"
            ),
            "district": row.get(
                "District"
            ),
            "category": row.get(
                "Category"
            ),
            "description": row.get(
                "Description"
            ),
            "cost_m_pkr": float(
                row["Cost (M)"]
            ),
            "category_average_m_pkr": (
                round(
                    float(
                        row["category_avg"]
                    ),
                    2
                )
                if pd.notna(
                    row["category_avg"]
                )
                else None
            ),
        })

    return {
        "agent": "Finance Agent",
        "total_not_started_budget_m_pkr": round(
            float(total_budget),
            2
        ),
        "budget_limit_m_pkr": 2000,
        "top_candidates": results,
    }


# ==================================================
# DELIVERY AGENT ANALYSIS
# ==================================================

def delivery_review(limit=10):
    """
    Delivery Agent:
    Check whether Not Started projects
    have the basic accountability/execution
    information required to proceed.
    """

    projects = get_not_started()

    if projects.empty:
        return {
            "agent": "Delivery Agent",
            "projects": [],
        }

    results = []

    for _, row in projects.iterrows():

        contractor = row.get(
            "Contractor"
        )

        xen = row.get(
            "XEN Name"
        )

        xen_contact = row.get(
            "XEN Contact"
        )

        nits = row.get(
            "NITs"
        )

        missing = []

        if pd.isna(contractor) or not str(
            contractor
        ).strip():

            missing.append(
                "Contractor"
            )

        if pd.isna(xen) or not str(
            xen
        ).strip():

            missing.append(
                "XEN Name"
            )

        if pd.isna(xen_contact) or not str(
            xen_contact
        ).strip():

            missing.append(
                "XEN Contact"
            )

        results.append({
            "global_id": row.get(
                "Global ID"
            ),
            "district": row.get(
                "District"
            ),
            "category": row.get(
                "Category"
            ),
            "description": row.get(
                "Description"
            ),
            "cost_m_pkr": row.get(
                "Cost (M)"
            ),
            "contractor": contractor,
            "xen_name": xen,
            "xen_contact": xen_contact,
            "nits": nits,
            "missing_accountability": missing,
        })

    # Projects with fewer missing fields
    # are easier to move forward.
    results.sort(
        key=lambda x: len(
            x["missing_accountability"]
        )
    )

    return {
        "agent": "Delivery Agent",
        "top_candidates": results[:limit],
    }


# ==================================================
# EQUITY AGENT ANALYSIS
# ==================================================

def equity_review(limit=10):
    """
    Equity Agent:
    Examine how Not Started funding is
    distributed across districts and categories.
    """

    projects = get_not_started()

    if projects.empty:
        return {
            "agent": "Equity Agent",
            "districts": [],
            "categories": [],
        }

    projects["Cost (M)"] = pd.to_numeric(
        projects["Cost (M)"],
        errors="coerce"
    ).fillna(0)

    # ----------------------------------------------
    # District distribution
    # ----------------------------------------------

    district_summary = (
        projects
        .groupby("District")
        .agg(
            projects=("Global ID", "count"),
            budget_m_pkr=("Cost (M)", "sum")
        )
        .reset_index()
        .sort_values(
            "budget_m_pkr",
            ascending=True
        )
    )

    districts = []

    for _, row in district_summary.head(
        limit
    ).iterrows():

        districts.append({
            "district": row["District"],
            "not_started_projects": int(
                row["projects"]
            ),
            "not_started_budget_m_pkr": round(
                float(
                    row["budget_m_pkr"]
                ),
                2
            ),
        })

    # ----------------------------------------------
    # Category distribution
    # ----------------------------------------------

    category_summary = (
        projects
        .groupby("Category")
        .agg(
            projects=("Global ID", "count"),
            budget_m_pkr=("Cost (M)", "sum")
        )
        .reset_index()
        .sort_values(
            "budget_m_pkr",
            ascending=True
        )
    )

    categories = []

    for _, row in category_summary.head(
        limit
    ).iterrows():

        categories.append({
            "category": row["Category"],
            "not_started_projects": int(
                row["projects"]
            ),
            "not_started_budget_m_pkr": round(
                float(
                    row["budget_m_pkr"]
                ),
                2
            ),
        })

    return {
        "agent": "Equity Agent",
        "least_funded_districts": districts,
        "least_funded_categories": categories,
    }


# ==================================================
# BOARD INPUT
# ==================================================

def build_board_input():

    return {
        "finance": finance_review(),
        "delivery": delivery_review(),
        "equity": equity_review(),
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print("\nFINANCE REVIEW")
    print(finance_review())

    print("\nDELIVERY REVIEW")
    print(delivery_review())

    print("\nEQUITY REVIEW")
    print(equity_review())