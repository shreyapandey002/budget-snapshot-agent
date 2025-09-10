# budget_snapshot_agent.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
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
    # lowercase columns
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
        # allow 'total' or fallback
        raise ValueError("Missing required column: amount (or synonyms)")

    # if tax exists: add to amount (safe numeric)
    if "tax" in df.columns:
        df["tax"] = pd.to_numeric(df["tax"], errors="coerce").fillna(0.0)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0) + df["tax"]

    # parse date & amount
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # drop invalid rows
    df = df.dropna(subset=["date", "department", "amount"])
    # normalize text fields
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

    # ensure numeric and non-negative
    weekly_df["before_budget"] = pd.to_numeric(weekly_df["before_budget"], errors="coerce").fillna(0.0)
    weekly_df["after_budget"] = pd.to_numeric(weekly_df["after_budget"], errors="coerce").fillna(0.0)
    return weekly_df

# ---- Adjustment parsing / utilities ----
PCT_RE = re.compile(r"^([+-]?\s*\d+(\.\d+)?)\s*%?$")

def parse_change_to_factor(change: str) -> Optional[float]:
    """
    Accepts:
      "5%" -> 1.05
      "+5%" -> 1.05
      "-10%" -> 0.9
      "5" -> 1.05
      "-5" -> 0.95
    Returns factor (float) or None if cannot parse.
    """
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
    # bare number means percent (not decimal)
    return 1.0 + (num / 100.0)

def match_mask_for_domain(df: pd.DataFrame, domain: str) -> pd.Series:
    """Return a boolean mask of rows matching the domain: department OR expense_category matches exactly (case-insensitive)"""
    domain_clean = domain.strip().lower()
    mask_dept = df["department"].astype(str).str.lower() == domain_clean
    mask_cat = pd.Series(False, index=df.index)
    if "expense_category" in df.columns:
        mask_cat = df["expense_category"].astype(str).str.lower() == domain_clean
    return mask_dept | mask_cat

def apply_json_adjustments(weekly_df: pd.DataFrame, adjustments: List[Adjustment]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies adjustments (percentage factors) to weekly_df rows.
    Maintains total company spend via hybrid rebalance:
        - Rows matched by any adjustment are considered constrained (their after_budget are set by adj factors)
        - Remaining rows are scaled proportionally so total_after == total_before
    Returns (updated_df, warnings)
    """
    df = weekly_df.copy()
    df["after_budget"] = df["after_budget"].astype(float).copy()
    warnings = []
    constrained_mask = pd.Series(False, index=df.index)

    # apply per-adjustment
    for adj in adjustments or []:
        domain = (adj.domain or "").strip()
        change = (adj.change or "").strip()
        if not domain or not change:
            warnings.append(f"Skipping invalid adjustment (empty domain/change): {adj}")
            continue
        factor = parse_change_to_factor(change)
        if factor is None or not math.isfinite(factor):
            warnings.append(f"Could not parse change '{change}' for domain '{domain}'")
            continue

        mask = match_mask_for_domain(df, domain)
        if not mask.any():
            # If no exact match, attempt partial match (contains)
            domain_low = domain.lower()
            mask_partial = df["department"].astype(str).str.lower().str.contains(domain_low) | \
                           (df["expense_category"].astype(str).str.lower().str.contains(domain_low) if "expense_category" in df.columns else pd.Series(False, index=df.index))
            if mask_partial.any():
                mask = mask_partial
                warnings.append(f"No exact match for '{domain}'; applied to partial matches.")
            else:
                warnings.append(f"No rows matched domain '{domain}'. Skipping.")
                continue

        # apply factor
        df.loc[mask, "after_budget"] = df.loc[mask, "after_budget"].astype(float) * float(factor)
        constrained_mask = constrained_mask | mask

    # Now rebalance to preserve company total (if any constraints applied)
    total_before = df["before_budget"].sum()
    total_after_constrained = df.loc[constrained_mask, "after_budget"].sum()
    remaining_after = df.loc[~constrained_mask, "after_budget"].sum()

    # If there were no constraints, nothing to rebalance
    if constrained_mask.any():
        # If remaining rows are zero but company totals are mismatched -> warn and skip rebalance
        if remaining_after == 0:
            if not math.isclose(total_after_constrained, total_before, rel_tol=1e-9, abs_tol=1e-6):
                warnings.append("All rows were constrained or remaining budget is zero; cannot rebalance remaining rows to preserve total. Totals may not match.")
            # nothing to do
        else:
            # rebalancer ensures total_after = total_before
            rebalancer = (total_before - total_after_constrained) / remaining_after
            # if rebalancer negative or zero -> warn and skip scaling (avoids negative budgets)
            if not math.isfinite(rebalancer) or rebalancer <= 0:
                warnings.append(f"Computed rebalancer {rebalancer:.4f} invalid; skipping proportional rebalance to avoid negative budgets.")
            else:
                df.loc[~constrained_mask, "after_budget"] = df.loc[~constrained_mask, "after_budget"] * rebalancer

    # numeric safety: clamp negatives to 0
    df["after_budget"] = pd.to_numeric(df["after_budget"], errors="coerce").fillna(0.0)
    df.loc[df["after_budget"] < 0, "after_budget"] = 0.0

    return df, warnings

def summarize_spending(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["department"]
    if "expense_category" in df.columns:
        group_cols.append("expense_category")

    summary = df.groupby(group_cols)[["before_budget", "after_budget"]].sum().reset_index()
    # percent change safe computation
    def pct_change(before, after):
        if before == 0:
            # if both zero -> 0%; if before zero and after >0 -> inf, represent as None or large
            if after == 0:
                return 0.0
            return float("inf")
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

    # header function
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

def generate_budget_pdf(weekly_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Budget Snapshot Report", 0, 1, "C")

    # Weekly breakdown
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Weekly Budget Breakdown", 0, 1)

    headers = ["Department"]
    if "expense_category" in weekly_df.columns:
        headers.append("Category")
    headers += ["Week Range", "Before", "After", "% Change"]

    if "expense_category" in weekly_df.columns:
        col_widths = [40, 40, 45, 25, 25, 25]
    else:
        col_widths = [60, 50, 30, 30, 30]

    rows = []
    for _, r in weekly_df.iterrows():
        week_range = f"{r['week_start'].strftime('%Y-%m-%d')} to {r['week_end'].strftime('%Y-%m-%d')}"
        before = float(r["before_budget"])
        after = float(r["after_budget"])
        pct = 0.0 if before == 0 else ((after - before) / before) * 100.0
        vals = [r["department"]]
        if "expense_category" in weekly_df.columns:
            vals.append(r["expense_category"])
        vals += [week_range, f"{before:,.2f}", f"{after:,.2f}", f"{pct:.2f}%"]
        rows.append(vals)

    _paged_table_to_pdf(pdf, headers, rows, col_widths)

    # Department summary
    pdf.add_page()
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
    for _, r in summary_df.iterrows():
        before = float(r["before_budget"])
        after = float(r["after_budget"])
        pct = "∞" if before == 0 and after > 0 else 0.0 if before == 0 else round((after - before) / before * 100, 2)
        vals = [r["department"]]
        if "expense_category" in summary_df.columns:
            vals.append(r["expense_category"])
        vals += [f"{before:,.2f}", f"{after:,.2f}", f"{pct}%" if isinstance(pct, float) else str(pct)]
        rows.append(vals)

    _paged_table_to_pdf(pdf, headers, rows, col_widths)
    pdf.output(output_path)

# -------------------------------
# Simple instruction parser for free-text
# -------------------------------
def parse_user_instructions(text: str) -> List[Adjustment]:
    """
    Very lightweight parser:
      - splits by 'and' or comma
      - looks for patterns like 'reduce X by 5%' / 'increase marketing by 10%'
      - fallback: picks tokens where a noun and percent appear
    """
    if not text or not text.strip():
        return []
    parts = re.split(r",|\band\b", text, flags=re.IGNORECASE)
    adjustments = []
    for p in parts:
        p_str = p.strip()
        # look for +/- percent
        m = re.search(r"(reduce|decrease|cut|lower)\s+([a-zA-Z &\-]+?)\s+by\s+([+-]?\d+(\.\d+)?\s*%?)", p_str, flags=re.IGNORECASE)
        if m:
            domain = m.group(2).strip()
            change = "-" + m.group(3).strip().lstrip("+-")
            adjustments.append(Adjustment(domain=domain, change=change))
            continue
        m = re.search(r"(increase|boost|raise|grow|expand)\s+([a-zA-Z &\-]+?)\s+by\s+([+-]?\d+(\.\d+)?\s*%?)", p_str, flags=re.IGNORECASE)
        if m:
            domain = m.group(2).strip()
            change = m.group(3).strip()
            adjustments.append(Adjustment(domain=domain, change=change))
            continue
        # fallback: find something like "marketing 10%" or "travel -5%"
        m2 = re.search(r"([a-zA-Z &\-]+?)\s*([+-]?\d+(\.\d+)?\s*%?)$", p_str)
        if m2:
            domain = m2.group(1).strip()
            change = m2.group(2).strip()
            adjustments.append(Adjustment(domain=domain, change=change))
            continue
    return adjustments

# -------------------------------
# Routes
# -------------------------------
@app.post("/generate-budget")
async def generate_budget(file: UploadFile = File(...), instructions: Optional[str] = None, adjustments: Optional[List[Adjustment]] = Body(None)):
    # validate suffix
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_SUFFIXES}")

    # small file size guard (if file.size header not provided we read, but cap)
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file too large.")
    # write to temp
    fd, tmp_file_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_file_path, "wb") as f:
            f.write(content)
        # process in threadpool
        def process_and_generate():
            # read file robustly
            if suffix == ".csv":
                raw = pd.read_csv(tmp_file_path)
            else:
                raw = pd.read_excel(tmp_file_path)

            df = normalize_columns(raw)
            weekly_df = aggregate_weekly(df)

            # determine adjustments: explicit adjustments param > request body 'adjustments' (we accept both), then instructions
            adjustments_list = adjustments or []
            if instructions and instructions.strip():
                adjustments_from_text = parse_user_instructions(instructions)
                # append parsed ones (but don't duplicate)
                for a in adjustments_from_text:
                    if not any((a.domain.lower() == ex.domain.lower() and a.change == ex.change) for ex in adjustments_list):
                        adjustments_list.append(a)

            warnings = []
            if adjustments_list:
                weekly_df, warn = apply_json_adjustments(weekly_df, adjustments_list)
                warnings.extend(warn)

            summary_df = summarize_spending(weekly_df)
            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(weekly_df, summary_df, tmp_pdf_file)
            return tmp_pdf_file, warnings

        tmp_pdf_file, warnings = await run_in_threadpool(process_and_generate)
        response = {"status": "success", "download_link": f"/download/{os.path.basename(tmp_pdf_file)}"}
        if warnings:
            response["warnings"] = warnings
        return response
    finally:
        try:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        except Exception:
            logger.exception("Failed to remove temp upload file")

@app.post("/generate-budget-url")
async def generate_budget_url(request: BudgetRequest):
    if not request.file_url:
        raise HTTPException(status_code=400, detail="file_url is required")
    # fetch securely
    try:
        resp = requests.get(request.file_url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Could not download file: HTTP {resp.status_code}")

    # check content length (if provided)
    cl = resp.headers.get("Content-Length")
    if cl and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Remote file too large")

    # write to tmp
    fd, tmp_file_path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        with open(tmp_file_path, "wb") as f:
            f.write(resp.content)

        def process_and_generate():
            raw = pd.read_excel(tmp_file_path)
            df = normalize_columns(raw)
            weekly_df = aggregate_weekly(df)

            # parse instructions into adjustments if present
            adjustments_list = request.adjustments or []
            if request.instructions:
                parsed = parse_user_instructions(request.instructions)
                for a in parsed:
                    if not any((a.domain.lower() == ex.domain.lower() and a.change == ex.change) for ex in adjustments_list):
                        adjustments_list.append(a)

            warnings = []
            if adjustments_list:
                weekly_df, warn = apply_json_adjustments(weekly_df, adjustments_list)
                warnings.extend(warn)

            summary_df = summarize_spending(weekly_df)
            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(weekly_df, summary_df, tmp_pdf_file)
            return tmp_pdf_file, warnings

        tmp_pdf_file, warnings = await run_in_threadpool(process_and_generate)
        response = {"status": "success", "download_link": f"/download/{os.path.basename(tmp_pdf_file)}"}
        if warnings:
            response["warnings"] = warnings
        return response
    finally:
        try:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        except Exception:
            logger.exception("Failed to remove temp download file")

@app.get("/download/{file_name}")
def download_file(file_name: str):
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="Budget_Snapshot.pdf", media_type="application/pdf")
    return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})
