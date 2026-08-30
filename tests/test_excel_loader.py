from src.ingestion.excel_loader import load_projects


df = load_projects("data/Projects.xlsx")

print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())