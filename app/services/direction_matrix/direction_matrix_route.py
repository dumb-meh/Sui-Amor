"""
Direction Matrix admin endpoint.

POST /admin/upload-direction-matrix
    Accepts an Excel file (.xlsx / .xls) or a CSV file.
    Parses the sheet containing the direction rules (auto-detected by its
    required column headers), extracts only the columns needed for the
    direction lookup, and saves the result to disk via direction_matrix.py.

GET /admin/direction-matrix/status
    Returns basic stats about the currently loaded matrix.
"""

import io
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/admin", tags=["Admin - Direction Matrix"])

# Required columns in the direction-matrix sheet.
REQUIRED_COLS = {
    "Rule_ID", "Q9_Goal_ID", "Goal_Target",
    "Q2_Answer_ID", "Q2_Answer",
    "Q8_State_ID", "Q8_State",
    "Q10_Obstacle_ID", "Q10_Obstacle",
    "Energy_Level", "Calming_Score", "Energizing_Score", "Neutral_Score",
    "Direction_Result", "Descriptor", "Physical_Blend_ID",
}

# Columns we actually persist (everything needed for lookup + useful metadata)
KEEP_COLS = [
    "Rule_ID", "Q9_Goal_ID", "Goal_Target",
    "Q2_Answer_ID", "Q2_Answer",
    "Q8_State_ID", "Q8_State",
    "Q10_Obstacle_ID", "Q10_Obstacle",
    "Energy_Level", "Calming_Score", "Energizing_Score", "Neutral_Score",
    "Direction_Result", "Descriptor", "Physical_Blend_ID",
]


def _parse_sheet(content: bytes, filename: str) -> list[dict]:
    """
    Parse an uploaded Excel or CSV file and return the direction matrix rows.

    For Excel files: scans every sheet and picks the first one that contains
    all REQUIRED_COLS headers.
    For CSV files: parses directly and validates headers.

    Returns a list of dicts — one per data row — with only KEEP_COLS retained.
    """
    try:
        import pandas as pd
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pandas is not installed. Add 'pandas openpyxl' to requirements.txt.",
        )

    lower_name = filename.lower()

    if lower_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content), dtype=str)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"CSV is missing required columns: {sorted(missing)}",
            )
        df = df[KEEP_COLS].fillna("")
        return df.to_dict(orient="records")

    elif lower_name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(content))
        target_df = None
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name, dtype=str)
            if REQUIRED_COLS.issubset(set(df.columns)):
                target_df = df
                break

        if target_df is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No sheet found containing all required columns. "
                    f"Required: {sorted(REQUIRED_COLS)}. "
                    f"Sheets in file: {xl.sheet_names}"
                ),
            )

        target_df = target_df[KEEP_COLS].fillna("")
        # Drop completely empty rows (no Goal_Target)
        target_df = target_df[target_df["Goal_Target"].str.strip() != ""]
        return target_df.to_dict(orient="records")

    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a .xlsx, .xls, or .csv file.",
        )


@router.post("/upload-direction-matrix")
async def upload_direction_matrix(file: UploadFile = File(...)):
    """
    Upload the direction matrix Excel/CSV file.

    The endpoint auto-detects which sheet (in Excel) contains the direction
    rules by looking for the required column headers, parses it, and saves the
    data to disk. The affirmation service picks up the new data on its next
    call without requiring a server restart.

    Accepted formats: .xlsx, .xls, .csv
    """
    from app.utils.direction_matrix import save_matrix

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    rows = _parse_sheet(content, file.filename)

    if not rows:
        raise HTTPException(
            status_code=422,
            detail="File was parsed successfully but contained no data rows.",
        )

    save_matrix(rows)

    # Compute a quick direction distribution summary for confirmation
    from collections import Counter
    directions = Counter(r.get("Direction_Result", "Unknown") for r in rows)

    return JSONResponse(
        status_code=200,
        content={
            "message": "Direction matrix uploaded and saved successfully.",
            "total_rows": len(rows),
            "direction_distribution": dict(directions),
            "unique_goals": len({r.get("Goal_Target", "") for r in rows}),
        },
    )


@router.get("/direction-matrix/status")
async def direction_matrix_status():
    """
    Return stats about the currently loaded direction matrix.
    Uses the in-memory index — no file read needed.
    """
    from app.utils.direction_matrix import get_stats, MATRIX_FILE

    if not MATRIX_FILE.exists():
        return {
            "loaded": False,
            "message": "No direction matrix has been uploaded yet.",
        }

    stats = get_stats()
    return {
        "loaded": True,
        "total_indexed_rules": stats["indexed_rules"],
        "direction_distribution": stats["direction_distribution"],
        "unique_goals": stats["unique_goals"],
        "file_path": str(MATRIX_FILE),
    }
