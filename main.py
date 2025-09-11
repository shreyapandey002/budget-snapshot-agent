# budget_snapshot_agent.py
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
import math
import re

# ----- Config -----
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
REQUEST_TIMEOUT = 10  # seconds for requests.get
ALLOWED_SUFFIXES = {".xls", ".xlsx", ".csv"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("budget_snapshot_agent")

app = FastAPI(title="Budget Snapshot Agent (Robust)")

# -------------------------------
# Pydantic models
# -------------------------------
class Adjustment(BaseModel):
    domain: str = Field(..., description="department name or expense_category name (case-insensitive)")
    change: str = Field(..., description="percent like '5%', '-10%', '+5' or numeric interpreted as percent")

class Instruction(BaseModel):
    action: str  # "adjust", "merge", "remove", "allocate", "headcount"
    domain: Optional[str] = None
    target: Optional[str] = None   # for merge target
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

    # required checks
    if "date" not in df.columns:
        raise ValueError("Missing required column: date")
    if "department" not in df.columns:
        raise ValueError("Missing required column: department")
    if "amount" not in df.columns:
        raise ValueError("Missing required column: amount (or synonyms)")

    # if tax exists: add to amount
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

def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["week_end"] = df["week_start"] + timedelta(days=6)
    group_cols = ["department", "week_start", "week_end"]
    if "expense_category" in df.columns:
        group_cols.insert(1, "expense_category")

    weekly_df = df.groupby(group_cols)["amount"].sum().reset_index()
    weekly_df = weekly_df.rename(columns={"amount": "before_budget"})
    weekly_df["after_budget"] = weekly_df["before_budget"].astype(float).copy()
    weekly_df["before_budget"] = pd.to_numeric(weekly_df["before_budget"], errors="coerce").fillna(0.0)
    weekly_df["after_budget"] = pd.to_numeric(weekly_df["after_budget"], errors="coerce").fillna(0.0)
    return weekly_df

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

def apply_instructions(weekly_df: pd.DataFrame, instructions: List[Instruction]) -> Tuple[pd.DataFrame, List[str]]:
    df = weekly_df.copy()
    warnings = []
    constrained_mask = pd.Series(False, index=df.index)

    # Merge
    for instr in [i for i in instructions if i.action == "merge"]:
        mask_src = match_mask_for_domain(df, instr.domain)
        mask_tgt = match_mask_for_domain(df, instr.target) if instr.target else pd.Series(False, index=df.index)
        if not mask_src.any():
            warnings.append(f"No rows matched to merge from '{instr.domain}'")
            continue
        if not mask_tgt.any():
            warnings.append(f"No rows matched to merge into '{instr.target}', skipping merge")
            continue
        
        # For each category under source, merge into target
        categories = df.loc[mask_src, "expense_category"].unique() if "expense_category" in df.columns else [None]
        for cat in categories:
            if cat is None:
                src_rows = df[mask_src]
                tgt_rows = df[mask_tgt]
            else:
                src_rows = df[mask_src & (df["expense_category"] == cat)]
                tgt_rows = df[mask_tgt & (df["expense_category"] == cat)]
            if src_rows.empty:
                continue
            
            if not tgt_rows.empty:
                # Add into existing target row
                df.loc[tgt_rows.index, ["before_budget", "after_budget"]] += src_rows[["before_budget", "after_budget"]].sum()
            else:
               # Move source row(s) to target dept
                moved = src_rows.copy()
                moved.loc[:, "department"] = instr.target
                df = pd.concat([df, moved], ignore_index=True)
        # Drop all source rows
        df = df.loc[~mask_src].reset_index(drop=True)

    # Remove
    for instr in [i for i in instructions if i.action == "remove"]:
        mask = match_mask_for_domain(df, instr.domain)
        if not mask.any():
            warnings.append(f"No rows matched to remove '{instr.domain}'")
            continue
        df = df.loc[~mask].reset_index(drop=True)

    # Allocate
    total_before = df["before_budget"].sum()
    for instr in [i for i in instructions if i.action == "allocate"]:
        mask = match_mask_for_domain(df, instr.domain)
        if not mask.any():
            warnings.append(f"No rows matched to allocate to '{instr.domain}'")
            continue
        target_share = (instr.percent / 100.0) * total_before
        df.loc[mask, "after_budget"] = target_share
        constrained_mask = constrained_mask | mask

    # Headcount + Adjust
    for instr in [i for i in instructions if i.action in ["headcount", "adjust"]]:
        factor = parse_change_to_factor(instr.change)
        if factor is None:
            warnings.append(f"Invalid change '{instr.change}' for {instr.domain}")
            continue
        mask = match_mask_for_domain(df, instr.domain)
        if not mask.any():
            warnings.append(f"No rows matched for {instr.action} '{instr.domain}'")
            continue
        df.loc[mask, "after_budget"] *= factor
        constrained_mask = constrained_mask | mask

    # Rebalance
    total_after_constrained = df.loc[constrained_mask, "after_budget"].sum()
    remaining_after = df.loc[~constrained_mask, "after_budget"].sum()

    if constrained_mask.any() and remaining_after > 0:
        rebalancer = (total_before - total_after_constrained) / remaining_after
        if math.isfinite(rebalancer) and rebalancer > 0:
            df.loc[~constrained_mask, "after_budget"] *= rebalancer
        else:
            warnings.append("Invalid rebalancer, skipped proportional scaling")

    df["after_budget"] = df["after_budget"].clip(lower=0)
    return df, warnings

def summarize_spending(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["department"]
    if "expense_category" in df.columns:
        group_cols.append("expense_category")
    summary = df.groupby(group_cols)[["before_budget", "after_budget"]].sum().reset_index()
    def pct_change(before, after):
        if before == 0:
            return 0.0 if after == 0 else float("inf")
        return round((after - before) / before * 100, 2)
    summary["percent_change"] = summary.apply(lambda r: pct_change(r["before_budget"], r["after_budget"]), axis=1)
    return summary

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

def generate_budget_pdf(weekly_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str, instructions_text: str = "", warnings: List[str] = []):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Budget Snapshot Report", 0, 1, "C")

    # Instructions + Warnings at top
    pdf.set_font("Arial", "B", 12)
    if instructions_text:
        pdf.multi_cell(0, 6, f"Instructions:\n{instructions_text}")
    if warnings:
        pdf.set_font("Arial", "I", 11)
        pdf.multi_cell(0, 6, "Warnings:\n" + "\n".join(warnings))
    pdf.ln(5)

    # Department Summary 
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Department Summary", 0, 1)

    headers = ["Department"]
    if "expense_category" in summary_df.columns:
        headers.append("Category")
    headers += ["Total Before", "Total After", "% Change"]

    if "expense_category" in summary_df.columns:
        col_widths = [50, 40, 35, 35, 30]
    else:
        col_widths = [60, 40, 40, 40]

    rows = []
    total_before_sum = 0.0
    total_after_sum = 0.0

    for _, r in summary_df.iterrows():
        before = float(r["before_budget"])
        after = float(r["after_budget"])
        pct = "∞" if before == 0 and after > 0 else 0.0 if before == 0 else round((after - before) / before * 100, 2)
        vals = [r["department"]]
        if "expense_category" in summary_df.columns:
            vals.append(r["expense_category"])
        vals += [f"{before:,.2f}", f"{after:,.2f}", f"{pct}%" if isinstance(pct, float) else str(pct)]
        rows.append(vals)

        total_before_sum += before
        total_after_sum += after

    # --- Add Grand Total row ---
    pct_total = "∞" if total_before_sum == 0 and total_after_sum > 0 else 0.0 if total_before_sum == 0 else round((total_after_sum - total_before_sum) / total_before_sum * 100, 2)
    total_row = ["GRAND TOTAL"]
    if "expense_category" in summary_df.columns:
        total_row.append("")  # blank for category column
    total_row += [
        f"{total_before_sum:,.2f}",
        f"{total_after_sum:,.2f}",
        f"{pct_total}%" if isinstance(pct_total, float) else str(pct_total),
    ]
    rows.append(total_row)
    
    _paged_table_to_pdf(pdf, headers, rows, col_widths)
    pdf.output(output_path)

# -------------------------------
# Simple instruction parser for free-text
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

        # Merge
        m = re.search(r"merge\s+([a-zA-Z &\-]+)\s+into\s+([a-zA-Z &\-]+)", p_str, flags=re.IGNORECASE)
        if m:
            instructions.append(Instruction(action="merge", domain=m.group(1).strip(), target=m.group(2).strip()))
            continue

        # Remove
        m = re.search(r"(remove|eliminate|delete)\s+([a-zA-Z &\-]+)", p_str, flags=re.IGNORECASE)
        if m:
            instructions.append(Instruction(action="remove", domain=m.group(2).strip()))
            continue

        # Allocate
        m = re.search(r"allocate\s+(\d+(\.\d+)?)%\s+(of\s+(the\s+)?)?(total\s+budget|budget)\s+(to|for)\s+([a-zA-Z &\-]+)", p_str, flags=re.IGNORECASE)
        if m:
            instructions.append(Instruction(action="allocate", domain=m.group(7).strip(), percent=float(m.group(1))))
            continue

        # Headcount
        m = re.search(r"(increase|decrease|reduce)\s+([a-zA-Z &\-]+)\s+headcount\s+by\s+([+-]?\d+(\.\d+)?\s*%?)", p_str, flags=re.IGNORECASE)
        if m:
            change = m.group(3).strip()
            if m.group(1).lower() in ["decrease", "reduce"] and not change.startswith("-"):
                change = "-" + change
            instructions.append(Instruction(action="headcount", domain=m.group(2).strip(), change=change))
            continue

        # Adjust
        m = re.search(r"(increase|decrease|reduce)\s+([a-zA-Z &\-]+)\s+by\s+([+-]?\d+(\.\d+)?\s*%?)", p_str, flags=re.IGNORECASE)
        if m:
            change = m.group(3).strip()
            if m.group(1).lower() in ["decrease", "reduce"] and not change.startswith("-"):
                change = "-" + change
            instructions.append(Instruction(action="adjust", domain=m.group(2).strip(), change=change))
            continue

        # Bare fallback
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
        with open(tmp_file_path, "wb") as f:
            f.write(content)

        def process_and_generate():
            raw = pd.read_csv(tmp_file_path) if suffix == ".csv" else pd.read_excel(tmp_file_path)
            df = normalize_columns(raw)
            weekly_df = aggregate_weekly(df)

            instructions_list: List[Instruction] = []
            if instructions and instructions.strip():
                instructions_list.extend(parse_user_instructions(instructions))
            if adjustments:
                for a in adjustments:
                    instructions_list.append(Instruction(action="adjust", domain=a.domain, change=a.change))

            warnings = []
            if instructions_list:
                weekly_df, warn = apply_instructions(weekly_df, instructions_list)
                warnings.extend(warn)

            summary_df = summarize_spending(weekly_df)
            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(weekly_df, summary_df, tmp_pdf_file, instructions or "", warnings)
            return tmp_pdf_file, warnings

        tmp_pdf_file, warnings = await run_in_threadpool(process_and_generate)
        response = {"status": "success", "download_link": f"https://budget-snapshot-agent.onrender.com/download/{os.path.basename(tmp_pdf_file)}"}
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
        with open(tmp_file_path, "wb") as f:
            f.write(resp.content)

        def process_and_generate():
            raw = pd.read_excel(tmp_file_path)
            df = normalize_columns(raw)
            weekly_df = aggregate_weekly(df)

            instructions_list: List[Instruction] = []
            if request.instructions and request.instructions.strip():
                instructions_list.extend(parse_user_instructions(request.instructions))
            if request.adjustments:
                for a in request.adjustments:
                    instructions_list.append(Instruction(action="adjust", domain=a.domain, change=a.change))

            warnings = []
            if instructions_list:
                weekly_df, warn = apply_instructions(weekly_df, instructions_list)
                warnings.extend(warn)

            summary_df = summarize_spending(weekly_df)
            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(weekly_df, summary_df, tmp_pdf_file, request.instructions or "", warnings)
            return tmp_pdf_file, warnings

        tmp_pdf_file, warnings = await run_in_threadpool(process_and_generate)
        response = {"status": "success", "download_link": f"https://budget-snapshot-agent.onrender.com/download/{os.path.basename(tmp_pdf_file)}"}
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
    return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})
