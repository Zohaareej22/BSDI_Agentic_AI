from src.tools.audit_tools import (
    audit_missing_work_started,
    audit_high_cost_no_contractor,
    audit_not_started_budget_by_district,
    audit_category_cost_outliers,
    audit_nits_no_in_progress,
)


print("\n" + "=" * 60)
print("AUDIT 1 — IN PROGRESS WITHOUT WORK STARTED")
print("=" * 60)

result = audit_missing_work_started()

print("Count:", result["count"])

for row in result["findings"][:3]:
    print(row)


print("\n" + "=" * 60)
print("AUDIT 2 — HIGH COST WITHOUT CONTRACTOR")
print("=" * 60)

result = audit_high_cost_no_contractor()

print(
    "Top 10% cost threshold:",
    result["threshold_m_pkr"],
    "M PKR",
)

print("Count:", result["count"])

for row in result["findings"][:3]:
    print(row)


print("\n" + "=" * 60)
print("AUDIT 3 — NOT STARTED BUDGET BY DISTRICT")
print("=" * 60)

result = audit_not_started_budget_by_district()

print("Count:", result["count"])

for row in result["findings"][:5]:
    print(row)


print("\n" + "=" * 60)
print("AUDIT 4 — CATEGORY COST OUTLIERS")
print("=" * 60)

result = audit_category_cost_outliers()

print("Count:", result["count"])

for row in result["findings"][:5]:
    print(row)


print("\n" + "=" * 60)
print("AUDIT 5 — NITS = NO BUT IN PROGRESS")
print("=" * 60)

result = audit_nits_no_in_progress()

print("Count:", result["count"])

for row in result["findings"][:5]:
    print(row)