from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import tempfile
import os
from fpdf import FPDF
import uuid
from datetime import timedelta
from starlette.concurrency import run_in_threadpool
import requests

app = FastAPI(title="Budget Snapshot Agent")

# -------------------------------
# Pydantic models
# -------------------------------
class Adjustment(BaseModel):
    domain: str
    change: str

class BudgetRequest(BaseModel):
    file_url: Optional[str] = None
    instructions: Optional[str] = ""
    adjustments: Optional[List[Adjustment]] = []

# -------------------------------
# Helpers (same as tumhare code)
# -------------------------------
def normalize_columns(df: pd.DataFrame):
    df.columns = [c.lower().strip() for c in df.columns]
    if "date" not in df.columns or "department" not in df.columns or "amount" not in df.columns:
        raise ValueError("Missing required columns (date, department, amount)")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "department", "amount"])
    return df

def aggregate_weekly(df: pd.DataFrame):
    df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["week_end"] = df["week_start"] + timedelta(days=6)
    weekly_df = df.groupby(["department", "week_start", "week_end"])["amount"].sum().reset_index()
    weekly_df.rename(columns={"amount": "before_budget"}, inplace=True)
    weekly_df["after_budget"] = weekly_df["before_budget"].copy()
    return weekly_df

def apply_json_adjustments(df: pd.DataFrame, adjustments: List[Adjustment]):
    for adj in adjustments:
        dept = adj.domain.strip().lower()
        change = adj.change.strip()
        if not dept or not change:
            continue
        if "%" in change:
            num = float(change.replace("%", "").replace("+", "").replace("-", ""))
            factor = 1 + (num / 100) if change.startswith("+") else 1 - (num / 100)
        else:
            try:
                num = float(change)
                factor = 1 + (num / 100)
            except:
                continue
        mask = df["department"].str.lower() == dept
        df.loc[mask, "after_budget"] *= factor
    return df

def summarize_department_spending(df: pd.DataFrame):
    summary = df.groupby("department")[["before_budget", "after_budget"]].sum().reset_index()
    summary["percent_change"] = ((summary["after_budget"] - summary["before_budget"]) / summary["before_budget"] * 100).round(2)
    return summary

def generate_budget_pdf(weekly_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Budget Snapshot Report", 0, 1, "C")
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
        values = [row["department"], week_range, f"{row['before_budget']:.2f}", f"{row['after_budget']:.2f}", f"{percent_change:.2f}%"]
        for v, w in zip(values, col_widths):
            pdf.cell(w, 8, str(v), 1)
        pdf.ln()

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
        values = [row["department"], f"{row['before_budget']:.2f}", f"{row['after_budget']:.2f}", f"{row['percent_change']:.2f}%"]
        for v, w in zip(values, col_widths):
            pdf.cell(w, 8, str(v), 1)
        pdf.ln()
    pdf.output(output_path)

# -------------------------------
# Route 1: File upload (form-data)
# -------------------------------
@app.post("/generate-budget")
async def generate_budget(file: UploadFile = File(...), request: Optional[str] = None):
    suffix = os.path.splitext(file.filename)[1]
    if suffix.lower() not in [".xls", ".xlsx"]:
        raise HTTPException(status_code=400, detail="Only Excel files are supported.")

    fd, tmp_file_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    content = await file.read()
    with open(tmp_file_path, "wb") as f:
        f.write(content)

    def process_and_generate():
        df = pd.read_excel(tmp_file_path)
        df = normalize_columns(df)
        weekly_df = aggregate_weekly(df)
        # TODO: parse request JSON string if needed
        summary_df = summarize_department_spending(weekly_df)
        tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
        generate_budget_pdf(weekly_df, summary_df, tmp_pdf_file)
        return tmp_pdf_file

    try:
        tmp_pdf_file = await run_in_threadpool(process_and_generate)
        return {"status": "success", "download_link": f"http://127.0.0.1:8000/download/{os.path.basename(tmp_pdf_file)}"}
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# -------------------------------
# Route 2: File URL (JSON body)
# -------------------------------
@app.post("/generate-budget-url")
async def generate_budget_url(request: BudgetRequest):
    try:
        # download from URL
        resp = requests.get(request.file_url)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download file from URL")

        fd, tmp_file_path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        with open(tmp_file_path, "wb") as f:
            f.write(resp.content)

        def process_and_generate():
            df = pd.read_excel(tmp_file_path)
            df = normalize_columns(df)
            weekly_df = aggregate_weekly(df)
            if request.adjustments:
                weekly_df = apply_json_adjustments(weekly_df, request.adjustments)
            summary_df = summarize_department_spending(weekly_df)
            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(weekly_df, summary_df, tmp_pdf_file)
            return tmp_pdf_file

        tmp_pdf_file = await run_in_threadpool(process_and_generate)
        return {"status": "success", "download_link": f"http://127.0.0.1:8000/download/{os.path.basename(tmp_pdf_file)}"}

    finally:
        if "tmp_file_path" in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# -------------------------------
# Download route
# -------------------------------
@app.get("/download/{file_name}")
def download_file(file_name: str):
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="Budget_Snapshot.pdf", media_type="application/pdf")
    return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})
