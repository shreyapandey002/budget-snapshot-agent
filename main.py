from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import pandas as pd
import tempfile
import os
from fpdf import FPDF
import uuid
from datetime import timedelta
from starlette.concurrency import run_in_threadpool
import requests
import logging
import re

# ----- Config -----
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
REQUEST_TIMEOUT = 10  # seconds for requests.get
ALLOWED_SUFFIXES = {".xls", ".xlsx", ".csv"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("budget_snapshot_agent")

app = FastAPI(title="Budget Snapshot Agent")

# -------------------------------
# Pydantic models
# -------------------------------
class Adjustment(BaseModel):
    domain: str = Field(..., description="department name or expense_category name (case-insensitive)")
    change: str = Field(..., description="percent like '5%', '-10%', '+5' or numeric interpreted as percent")

class Instruction(BaseModel):
    action: str  # "adjust", "remove", "allocate", "headcount"
    domain: Optional[str] = None
    target: Optional[str] = None   # for merge target (unused)
    change: Optional[str] = None   # "+10%", "-5%" for adjust/headcount
    percent: Optional[float] = None  # for allocate

class BudgetRequest(BaseModel):
    file_url: Optional[str] = None
    instructions: Optional[str] = ""
    adjustments: Optional[List[Adjustment]] = []

# -------------------------------
# Column synonyms & helpers
# -------------------------------
COLUMN_SYNONYMS = {
    "department": ["department", "company", "team", "function"],
    "amount": ["amount", "spend", "cost", "value", "expense", "total", "total_amount"],
    "tax": ["tax", "vat", "gst"],
    "expense_category": ["category", "expense_category", "item", "purpose"]
}

def find_column(df_cols, synonyms):
    for s in synonyms:
        if s in df_cols:
            return s
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    col_map = {}

    for canon, alts in COLUMN_SYNONYMS.items():
        found = find_column(df.columns, alts)
        if found:
            col_map[found] = canon

    df = df.rename(columns=col_map)

    if "date" not in df.columns:
        raise ValueError("Missing required column: date")
    if "department" not in df.columns:
        raise ValueError("Missing required column: department")
    if "amount" not in df.columns:
        raise ValueError("Missing required column: amount (or synonyms)")

    if "tax" in df.columns:
        df["tax"] = pd.to_numeric(df["tax"], errors="coerce").fillna(0.0)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0) + df["tax"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "department", "amount"])

    df["department"] = df["department"].astype(str).str.strip()
    if "expense_category" in df.columns:
        df["expense_category"] = df["expense_category"].astype(str).str.strip()

    return df

# -------------------------------
# Aggregation helpers
# -------------------------------
def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["month_name"] = df["month"].dt.strftime("%B %Y")
    
    group_cols = ["department", "month_name"]
    if "expense_category" in df.columns:
        group_cols.insert(1, "expense_category")
    
    monthly_df = df.groupby(group_cols)["amount"].sum().reset_index()
    monthly_df = monthly_df.rename(columns={"amount": "previous_year"})
    monthly_df["this_year"] = monthly_df["previous_year"].astype(float).copy()
    
    return monthly_df

# ---- Adjustment parsing / utilities ----
PCT_RE = re.compile(r"^([+-]?\s*\d+(\.\d+)?)\s*%?$")

def parse_change_to_factor(change: str) -> Optional[float]:
    if not isinstance(change, str):
        return None
    s = change.strip().replace(" ", "")
    m = PCT_RE.match(s)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except Exception:
        return None
    return 1.0 + (num / 100.0)

def match_mask_for_domain(df: pd.DataFrame, domain: str) -> pd.Series:
    domain_clean = domain.strip().lower()
    mask_dept = df["department"].astype(str).str.lower() == domain_clean
    mask_cat = pd.Series(False, index=df.index)
    if "expense_category" in df.columns:
        mask_cat = df["expense_category"].astype(str).str.lower() == domain_clean
    return mask_dept | mask_cat

# -------------------------------
# Yearly adjustments applied here
# -------------------------------
def summarize_spending(df: pd.DataFrame, instructions: List[Instruction]) -> Tuple[pd.DataFrame, List[str]]:
    warnings = []
    yearly = df.groupby("department")[["previous_year","this_year"]].sum().reset_index()

    # Apply department-level adjustments
    for instr in [i for i in instructions if i.action in ["adjust", "headcount"]]:
        factor = parse_change_to_factor(instr.change)
        if factor is None:
            warnings.append(f"Invalid change '{instr.change}' for {instr.domain}")
            continue
        mask = yearly["department"].astype(str).str.lower() == instr.domain.lower()
        if not mask.any():
            warnings.append(f"No department matched '{instr.domain}' in yearly summary")
            continue
        yearly.loc[mask, "this_year"] *= factor

    return yearly, warnings

# -------------------------------
# PDF / Reporting
# -------------------------------
def _paged_table_to_pdf(pdf: FPDF, headers: List[str], rows: List[List[str]], col_widths: List[int], font_size=9):
    pdf.set_font("Arial", "", font_size)
    line_h = font_size * 0.45 + 4
    max_lines_per_page = int((pdf.h - 40) / line_h)
    cur_line = 0

    def render_header():
        pdf.set_font("Arial", "B", font_size)
        for h, w in zip(headers, col_widths):
            pdf.cell(w, line_h, str(h), 1)
        pdf.ln()

    render_header()
    cur_line += 1
    pdf.set_font("Arial", "", font_size)

    for row in rows:
        if cur_line >= max_lines_per_page:
            pdf.add_page()
            render_header()
            cur_line = 1
        for v, w in zip(row, col_widths):
            pdf.cell(w, line_h, str(v), 1)
        pdf.ln()
        cur_line += 1

def generate_budget_pdf(df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str, instructions_text: str = "", warnings: List[str] = []):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Budget Snapshot Report", 0, 1, "C")

    pdf.set_font("Arial", "B", 12)
    if instructions_text:
        pdf.multi_cell(0, 6, f"Instructions:\n{instructions_text}")
    if warnings:
        pdf.set_font("Arial", "I", 11)
        pdf.multi_cell(0, 6, "Warnings:\n" + "\n".join(warnings))
    pdf.ln(5)

    # -----------------
    # Monthly Grand Totals
    # -----------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Grand Totals", 0, 1)
    monthly_total = df.groupby("month_name")[["previous_year", "this_year"]].sum().reset_index()
    headers = ["Month", "Previous Year", "This Year", "% Change"]
    col_widths = [60, 50, 50, 30]
    rows = []
    for _, r in monthly_total.iterrows():
        prev, this = r["previous_year"], r["this_year"]
        pct = round((this-prev)/prev*100,2) if prev != 0 else ("∞" if this>0 else 0)
        rows.append([r["month_name"], f"{prev:,.2f}", f"{this:,.2f}", f"{pct}%"])
    _paged_table_to_pdf(pdf, headers, rows, col_widths)

    # -----------------
    # Yearly Summary
    # -----------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Yearly Summary", 0, 1)
    yearly_summary = summary_df.copy()
    yearly_summary["percent_change"] = yearly_summary.apply(
        lambda r: round((r["this_year"]-r["previous_year"])/r["previous_year"]*100,2)
        if r["previous_year"]!=0 else ("∞" if r["this_year"]>0 else 0), axis=1)
    grand_total = yearly_summary[["previous_year","this_year"]].sum()
    headers = ["Department","Previous Year","This Year","% Change"]
    col_widths = [60,50,50,30]
    rows = [[r["department"], f"{r['previous_year']:,.2f}", f"{r['this_year']:,.2f}", f"{r['percent_change']}%"]
            for _,r in yearly_summary.iterrows()]
    rows.append([
        "GRAND TOTAL",
        f"{grand_total['previous_year']:,.2f}",
        f"{grand_total['this_year']:,.2f}",
        f"{round((grand_total['this_year']-grand_total['previous_year'])/grand_total['previous_year']*100,2) if grand_total['previous_year']!=0 else ('∞' if grand_total['this_year']>0 else 0)}%"
    ])
    _paged_table_to_pdf(pdf, headers, rows, col_widths)

    pdf.output(output_path)

# -------------------------------
# Instruction parser
# -------------------------------
def parse_user_instructions(text: str) -> List[Instruction]:
    if not text or not text.strip():
        return []
    parts = re.split(r",|\band\b", text, flags=re.IGNORECASE)
    instructions: List[Instruction] = []
    for p in parts:
        p_str = p.strip()
        if not p_str:
            continue

        m = re.search(r"(remove|eliminate|delete)\s+([a-zA-Z &\-]+)", p_str, flags=re.IGNORECASE)
        if m:
            instructions.append(Instruction(action="remove", domain=m.group(2).strip()))
            continue

        m = re.search(r"allocate\s+(\d+(\.\d+)?)%\s+(of\s+(the\s+)?)?(total\s+budget|budget)\s+(to|for)\s+([a-zA-Z &\-]+)", p_str, flags=re.IGNORECASE)
        if m:
            instructions.append(Instruction(action="allocate", domain=m.group(7).strip(), percent=float(m.group(1))))
            continue

        m = re.search(r"(increase|decrease|reduce)\s+([a-zA-Z &\-]+)\s+headcount\s+by\s+([+-]?\d+(\.\d+)?\s*%?)", p_str, flags=re.IGNORECASE)
        if m:
            change = m.group(3).strip()
            if m.group(1).lower() in ["decrease","reduce"] and not change.startswith("-"):
                change = "-"+change
            instructions.append(Instruction(action="headcount", domain=m.group(2).strip(), change=change))
            continue

        m = re.search(r"(increase|decrease|reduce)\s+([a-zA-Z &\-]+)\s+by\s+([+-]?\d+(\.\d+)?\s*%?)", p_str, flags=re.IGNORECASE)
        if m:
            change = m.group(3).strip()
            if m.group(1).lower() in ["decrease","reduce"] and not change.startswith("-"):
                change = "-"+change
            instructions.append(Instruction(action="adjust", domain=m.group(2).strip(), change=change))
            continue

        m = re.search(r"([a-zA-Z &\-]+)\s*([+-]?\d+(\.\d+)?\s*%?)$", p_str)
        if m:
            instructions.append(Instruction(action="adjust", domain=m.group(1).strip(), change=m.group(2).strip()))

    return instructions

# -------------------------------
# Routes
# -------------------------------
@app.post("/generate-budget")
async def generate_budget(file: UploadFile = File(...), instructions: Optional[str] = None, adjustments: Optional[List[Adjustment]] = Body(None)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file too large.")
    fd, tmp_file_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_file_path,"wb") as f:
            f.write(content)

        def process_and_generate():
            raw = pd.read_csv(tmp_file_path) if suffix==".csv" else pd.read_excel(tmp_file_path)
            df = normalize_columns(raw)
            monthly_df = aggregate_monthly(df)
            instructions_list: List[Instruction] = []
            if instructions and instructions.strip():
                instructions_list.extend(parse_user_instructions(instructions))
            if adjustments:
                for a in adjustments:
                    instructions_list.append(Instruction(action="adjust", domain=a.domain, change=a.change))
            warnings=[]
            summary_df, warn2 = summarize_spending(monthly_df, instructions_list)
            warnings.extend(warn2)
            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(monthly_df, summary_df, tmp_pdf_file, instructions or "", warnings)
            return tmp_pdf_file, warnings

        tmp_pdf_file, warnings = await run_in_threadpool(process_and_generate)
        response = {"status":"success", "download_link": f"https://budget-snapshot-agent.onrender.com/download/{os.path.basename(tmp_pdf_file)}"}
        if warnings:
            response["warnings"] = warnings
        return response
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.post("/generate-budget-url")
async def generate_budget_url(request: BudgetRequest):
    if not request.file_url:
        raise HTTPException(status_code=400, detail="file_url is required")
    try:
        resp = requests.get(request.file_url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Could not download file: HTTP {resp.status_code}")
    cl = resp.headers.get("Content-Length")
    if cl and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Remote file too large")

    fd, tmp_file_path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        with open(tmp_file_path,"wb") as f:
            f.write(resp.content)

        def process_and_generate():
            raw = pd.read_excel(tmp_file_path)
            df = normalize_columns(raw)
            monthly_df = aggregate_monthly(df)
            instructions_list: List[Instruction] = []
            if request.instructions and request.instructions.strip():
                instructions_list.extend(parse_user_instructions(request.instructions))
            if request.adjustments:
                for a in request.adjustments:
                    instructions_list.append(Instruction(action="adjust", domain=a.domain, change=a.change))
            warnings=[]
            summary_df, warn2 = summarize_spending(monthly_df, instructions_list)
            warnings.extend(warn2)
            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(monthly_df, summary_df, tmp_pdf_file, request.instructions or "", warnings)
            return tmp_pdf_file, warnings

        tmp_pdf_file, warnings = await run_in_threadpool(process_and_generate)
        response = {"status":"success", "download_link": f"https://budget-snapshot-agent.onrender.com/download/{os.path.basename(tmp_pdf_file)}"}
        if warnings:
            response["warnings"] = warnings
        return response
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.get("/download/{file_name}")
def download_file(file_name: str):
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="Budget_Snapshot.pdf", media_type="application/pdf")
    return JSONResponse(status_code=404, content={"status":"error","message":"File not found"})
