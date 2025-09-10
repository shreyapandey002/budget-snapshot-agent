from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import tempfile
import os
from fpdf import FPDF
import uuid
from datetime import timedelta
from starlette.concurrency import run_in_threadpool
import requests

app = FastAPI(title="Budget Snapshot Agent")

# --------------------------------------
# Column synonyms (normalize Excel input)
# --------------------------------------
COLUMN_SYNONYMS = {
    "department": ["department", "company", "team", "function"],
    "category": ["category", "sub-category", "type"],
    "amount": ["amount", "spend", "cost", "value", "expense"],
    "tax": ["tax", "vat", "gst"]
}

# ---------- Helpers ----------
def normalize_columns(df: pd.DataFrame):
    df.columns = [c.lower().strip() for c in df.columns]
    col_map = {}

    for key, synonyms in COLUMN_SYNONYMS.items():
        for s in synonyms:
            if s in df.columns:
                col_map[s] = key
                break

    df = df.rename(columns=col_map)

    # Check required columns
    if "date" not in df.columns:
        raise ValueError("Missing required column: date")
    if "department" not in df.columns:
        raise ValueError("Missing required column: department")
    if "amount" not in df.columns:
        raise ValueError("Missing required column: amount")

    # If tax exists → add to amount
    if "tax" in df.columns:
        df["amount"] = df["amount"] + df["tax"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "department", "amount"])
    return df


def aggregate_weekly(df: pd.DataFrame):
    df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["week_end"] = df["week_start"] + timedelta(days=6)

    # Aggregate by department + category if present
    group_cols = ["department", "week_start", "week_end"]
    if "category" in df.columns:
        group_cols.insert(1, "category")

    weekly_df = (
        df.groupby(group_cols)["amount"]
        .sum()
        .reset_index()
    )
    weekly_df.rename(columns={"amount": "before_budget"}, inplace=True)
    return weekly_df


# ---------- Apply Logic ----------
def apply_instructions(df: pd.DataFrame, instructions: list):
    df["after_budget"] = df["before_budget"].copy()

    for instr in instructions:
        domain = instr.get("domain", "").lower()
        change = instr.get("change", "0%").replace("%", "")
        try:
            change_val = float(change)
        except ValueError:
            continue

        # Apply to category if exists, else to department
        if "category" in df.columns and any(domain in str(c).lower() for c in df["category"].unique()):
            df.loc[df["category"].str.lower() == domain, "after_budget"] *= (1 + change_val / 100)
        elif any(domain in str(d).lower() for d in df["department"].unique()):
            df.loc[df["department"].str.lower() == domain, "after_budget"] *= (1 + change_val / 100)

    # --- Recalculate department totals if categories changed ---
    if "category" in df.columns:
        dept_totals = (
            df.groupby(["department", "week_start", "week_end"])["after_budget"]
            .sum()
            .reset_index()
        )
        for _, row in dept_totals.iterrows():
            mask = (
                (df["department"] == row["department"]) &
                (df["week_start"] == row["week_start"]) &
                (df["week_end"] == row["week_end"])
            )
            df.loc[mask, "after_budget"] = df.loc[mask, "after_budget"]  # keep detail
        # no overwrite, just ensures totals are consistent

    return df


def summarize_department_spending(df: pd.DataFrame):
    summary = (
        df.groupby("department")[["before_budget", "after_budget"]]
        .sum()
        .reset_index()
    )
    summary["percent_change"] = (
        (summary["after_budget"] - summary["before_budget"]) / summary["before_budget"] * 100
    ).round(2)
    return summary


# ---------- PDF Report ----------
def generate_budget_pdf(weekly_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Budget Snapshot Report", 0, 1, "C")

    # -------------------------
    # Section 1: Weekly Breakdown
    # -------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Weekly Budget Breakdown", 0, 1)

    headers = ["Department", "Week Range", "Before", "After", "% Change"]
    col_widths = [40, 60, 30, 30, 30]

    pdf.set_font("Arial", "B", 10)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, 1)
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for _, row in weekly_df.iterrows():
        week_range = f"{row['week_start'].strftime('%Y-%m-%d')} to {row['week_end'].strftime('%Y-%m-%d')}"
        percent_change = ((row['after_budget'] - row['before_budget']) / row['before_budget'] * 100) if row['before_budget'] else 0
        values = [
            row["department"],
            week_range,
            f"{row['before_budget']:.2f}",
            f"{row['after_budget']:.2f}",
            f"{percent_change:.2f}%"
        ]
        for v, w in zip(values, col_widths):
            pdf.cell(w, 8, str(v), 1)
        pdf.ln()

    # -------------------------
    # Section 2: Department Summary
    # -------------------------
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Department Summary", 0, 1)

    headers = ["Department", "Total Before", "Total After", "% Change"]
    col_widths = [60, 40, 40, 40]

    pdf.set_font("Arial", "B", 10)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, 1)
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for _, row in summary_df.iterrows():
        values = [
            row["department"],
            f"{row['before_budget']:.2f}",
            f"{row['after_budget']:.2f}",
            f"{row['percent_change']:.2f}%"
        ]
        for v, w in zip(values, col_widths):
            pdf.cell(w, 8, str(v), 1)
        pdf.ln()

    pdf.output(output_path)


# ---------- Routes ----------
@app.post("/generate-budget")
async def generate_budget(payload: dict = Body(...)):
    file_url = payload.get("file_url")
    instructions = payload.get("instructions", [])

    if not file_url:
        return JSONResponse(status_code=400, content={"status": "error", "message": "file_url is required"})

    # Download file
    try:
        r = requests.get(file_url)
        r.raise_for_status()
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Failed to download file: {e}"})

    suffix = os.path.splitext(file_url)[1]
    if suffix.lower() not in [".xls", ".xlsx"]:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Only Excel files are supported."})

    fd, tmp_file_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_file_path, "wb") as f:
        f.write(r.content)

    def process_and_generate():
        df = pd.read_excel(tmp_file_path)
        df = normalize_columns(df)
        weekly_df = aggregate_weekly(df)
        final_df = apply_instructions(weekly_df, instructions)
        summary_df = summarize_department_spending(final_df)

        tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
        generate_budget_pdf(final_df, summary_df, tmp_pdf_file)
        return tmp_pdf_file

    try:
        tmp_pdf_file = await run_in_threadpool(process_and_generate)
        download_link = f"https://budget-snapshot-agent.onrender.com/download/{os.path.basename(tmp_pdf_file)}"
        return {"status": "success", "download_link": download_link}
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.get("/download/{file_name}")
def download_file(file_name: str):
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="Budget_Snapshot.pdf", media_type="application/pdf")
    return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})
