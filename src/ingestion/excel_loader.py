from pathlib import Path
import pandas as pd


def load_projects(file_path: str | Path) -> pd.DataFrame:
    """
    Load the BSDI Projects Excel file.

    The actual column headers are on row 4,
    so the first 3 rows are skipped.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_excel(
        file_path,
        sheet_name="Projects List",
        skiprows=3
    )

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # Convert numeric columns
    if "Cost (M)" in df.columns:
        df["Cost (M)"] = pd.to_numeric(
            df["Cost (M)"],
            errors="coerce"
        )

    if "Progress %" in df.columns:
        df["Progress %"] = pd.to_numeric(
            df["Progress %"],
            errors="coerce"
        )

    return df