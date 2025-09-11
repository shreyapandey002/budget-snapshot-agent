# budget_snapshot_agent_fixed.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import pandas as pd
import tempfile
import os
from fpdf import FPDF
import uuid
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

app = FastAPI(title="Budget Snapshot Agent (fixed)")

# -------------------------------
# Pydantic models
# -------------------------------
class Adjustment(BaseModel):
    domain: str = Field(..., description="department name or expense_category (case-insensitive)")
    change: Optional[str] = Field(None, description="percent like '-10%', '+5%' or numeric interpreted as percent/value")
    # note: for allocate, use change as None and set percent in Instruction form below if needed
    percent: Optional[float] = None  # used for allocate if present

class BudgetRequest(BaseModel):
    file_url: Optional[str] = None
    instructions: Optional[str] = ""   # we'll ignore free-text instructions; kept for compatibility
    adjustments: Optional[List[Adjustment]] = []

# -------------------------------
# Column synonyms & helpers
# -------------------------------
COLUMN_SYNONYMS = {
    "department": ["department", "company", "team", "function"],
    "amount": ["amount", "spend", "cost", "value", "expense", "total", "total_amount"],
    "tax": ["tax", "vat", "gst"],
    "expense_category": ["category", "expense_category", "item", "purpose"],
    "date": ["date", "transaction_date", "txn_date"]
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

    # tax handling if present
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
    """
    Accepts '-10%', '+5%', '10' (interpreted as percent), '0.9' (interpreted as multiplier)
    Returns multiplier (1 + percent/100) or None for invalid.
    """
    if change is None:
        return None
    if isinstance(change, (int, float)):
        # numeric value interpreted as percent
        return 1.0 + (float(change) / 100.0)
    s = str(change).strip()
    # If it's already a multiplier like "0.9" treat as multiplier
    try:
        # but avoid interpreting "10" (should be percent). So only treat floats between 0 and 2 but no % sign?
        if (not s.endswith("%")) and ("." in s):
            v = float(s)
            # if v looks like a plausible multiplier (0.01-10), accept
            if 0.001 < v < 1000:
                return v if v < 10 else 1.0 + (v/100.0)
    except Exception:
        pass
    m = PCT_RE.match(s.replace(" ", ""))
    if not m:
        return None
    try:
        num = float(m.group(1))
    except Exception:
        return None
    return 1.0 + (num / 100.0)

def _normalize_for_match(s: str) -> str:
    """
    Normalize department/instruction domain for matching:
    - lowercased
    - remove words like 'budget', 'spend', 'expense', 'department'
    - remove punctuation
    - strip
    """
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\b(budget|spend|expense|expenses|department|dept|costs?)\b", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------------
# Core: compute dept factors and apply to monthly
# -------------------------------
def apply_adjustments_to_monthly(monthly_df: pd.DataFrame, adjustments: List[Adjustment]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    - monthly_df must have columns: department, (expense_category optional), month_name, previous_year, this_year
    - adjustments: list of Adjustment(domain, change, percent)
    Returns (monthly_adj_df, yearly_summary_df, warnings)
    """
    df = monthly_df.copy()
    warnings: List[str] = []

    # compute department-level previous totals
    dept_prev = df.groupby("department")["previous_year"].sum()
    # mapping normalized name -> actual department(s) (choose first)
    norm_to_dept = {}
    for dept in dept_prev.index:
        norm = _normalize_for_match(dept)
        if norm not in norm_to_dept:
            norm_to_dept[norm] = dept

    # init factors for each existing department
    factors = {dept: 1.0 for dept in dept_prev.index}
    grand_prev = float(dept_prev.sum())

    # We'll collect allocation rows to add later (for allocate into non-existing depts or zero-prev)
    allocation_rows = []

    # First pass: handle removes and adjusts (multiplicative)
    for adj in adjustments or []:
        domain_norm = _normalize_for_match(adj.domain)
        matched_dept = norm_to_dept.get(domain_norm)

        # If change is None but percent is provided -> treat as allocate
        is_allocate = (adj.percent is not None)

        if is_allocate:
            # Skip allocate for now — handle in next pass
            continue

        # Remove action: if change == 'remove' (we didn't model action field in Adjustment), user expects remove via domain?
        # We'll treat change == 'remove' or domain starts with 'remove ' as special — but since adjustments are JSON, removal likely not used.
        # For simplicity, if adj.change in ['remove','delete','eliminate'] treat as remove
        if isinstance(adj.change, str) and adj.change.strip().lower() in ("remove", "delete", "eliminate"):
            if matched_dept:
                factors[matched_dept] = 0.0
            else:
                # maybe removal intended for an expense_category — attempt to remove rows where expense_category matches
                if "expense_category" in df.columns:
                    cat_mask = df["expense_category"].astype(str).str.lower() == domain_norm
                    if cat_mask.any():
                        df = df.loc[~cat_mask].reset_index(drop=True)
                    else:
                        warnings.append(f"No department/category matched '{adj.domain}' for removal")
                else:
                    warnings.append(f"No department matched '{adj.domain}' for removal")
            continue

        # Adjust/headcount: percent or numeric (e.g., "-10%")
        factor = parse_change_to_factor(adj.change) if adj.change is not None else None
        if factor is None:
            warnings.append(f"Invalid change '{adj.change}' for '{adj.domain}' (skipped)")
            continue
        if matched_dept:
            factors[matched_dept] *= factor
        else:
            warnings.append(f"No department matched '{adj.domain}' (skipped)")

    # Second pass: handle allocations (adj.percent)
    for adj in adjustments or []:
        if adj.percent is None:
            continue
        domain_norm = _normalize_for_match(adj.domain)
        matched_dept = norm_to_dept.get(domain_norm)

        target_share = (adj.percent / 100.0) * grand_prev

        if matched_dept:
            prev_total = float(dept_prev.get(matched_dept, 0.0))
            if prev_total <= 0:
                # prev was zero: we cannot compute a multiplicative factor; create an allocation row
                allocation_rows.append({
                    "department": matched_dept,
                    "month_name": "(allocation)",
                    "previous_year": 0.0,
                    "this_year": target_share
                })
            else:
                # set factor so that dept_prev * factor == target_share
                factors[matched_dept] = target_share / prev_total
        else:
            # dept doesn't exist: create new allocation department row
            allocation_rows.append({
                "department": adj.domain.strip(),
                "month_name": "(allocation)",
                "previous_year": 0.0,
                "this_year": target_share
            })
            warnings.append(f"Allocated {adj.percent}% to new department '{adj.domain}' (created allocation row)")

    # Debug log factors
    logger.debug("Department adjustment factors: %s", factors)

    # Apply factors to monthly rows
    def _apply_factor(row):
        dept = row["department"]
        factor = factors.get(dept, 1.0)
        return row["previous_year"] * factor

    df["this_year"] = df.apply(_apply_factor, axis=1)

    # Append allocation rows if any
    if allocation_rows:
        alloc_df = pd.DataFrame(allocation_rows)
        # ensure columns match
        alloc_df["previous_year"] = alloc_df["previous_year"].astype(float)
        alloc_df["this_year"] = alloc_df["this_year"].astype(float)
        # If expense_category column exists, add NaN or default
        if "expense_category" in df.columns and "expense_category" not in alloc_df.columns:
            alloc_df["expense_category"] = ""
            # reorder to match df columns
            alloc_df = alloc_df[df.columns.intersection(alloc_df.columns).tolist() + ["previous_year","this_year"]]
        df = pd.concat([df, alloc_df], ignore_index=True, sort=False).fillna({"previous_year": 0.0, "this_year": 0.0})

    # Recompute yearly summary from adjusted monthly rows
    if "expense_category" in df.columns:
        yearly = df.groupby(["department"])[["previous_year", "this_year"]].sum().reset_index()
    else:
        yearly = df.groupby("department")[["previous_year", "this_year"]].sum().reset_index()

    # Ensure numeric types
    yearly["previous_year"] = yearly["previous_year"].astype(float)
    yearly["this_year"] = yearly["this_year"].astype(float)

    return df.reset_index(drop=True), yearly.reset_index(drop=True), warnings

# -------------------------------
# PDF / Reporting (keep style similar to your original)
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

def generate_budget_pdf(monthly_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str, instructions_text: str = "", warnings: List[str] = []):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Budget Snapshot Report", 0, 1, "C")

    if instructions_text or warnings:
        pdf.set_font("Arial", "", 11)
        if instructions_text:
            pdf.multi_cell(0, 6, f"Instructions:\n{instructions_text}")
        if warnings:
            pdf.multi_cell(0, 6, "Warnings:\n" + "\n".join(warnings))
        pdf.ln(5)

    # -------------------- Monthly Grand Totals --------------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Grand Totals", 0, 1)
    monthly_total = monthly_df.groupby("month_name")[["previous_year", "this_year"]].sum().reset_index()
    try:
        monthly_total["month_ts"] = pd.to_datetime(monthly_total["month_name"], format="%B %Y")
        monthly_total = monthly_total.sort_values("month_ts").drop(columns=["month_ts"])
    except Exception:
        monthly_total = monthly_total.sort_values("month_name")

    headers = ["Month", "Previous Year", "This Year", "% Change"]
    rows = []
    for _, r in monthly_total.iterrows():
        prev, this = r["previous_year"], r["this_year"]
        pct = round((this - prev) / prev * 100, 2) if prev != 0 else ("∞" if this > 0 else 0)
        rows.append([r["month_name"], f"{prev:,.2f}", f"{this:,.2f}", f"{pct}%"])
    col_widths = [60, 50, 50, 30]
    _paged_table_to_pdf(pdf, headers, rows, col_widths, font_size=9)

    # -------------------- Monthly Department Breakdown --------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Department Breakdown", 0, 1)
    headers = ["Month", "Department", "Previous Year", "This Year", "% Change"]

    dept_rows = []
    for _, r in monthly_df.iterrows():
        prev, this = r["previous_year"], r["this_year"]
        pct = round((this - prev) / prev * 100, 2) if prev != 0 else ("∞" if this > 0 else 0)
        dept_rows.append([r["month_name"], r["department"], f"{prev:,.2f}", f"{this:,.2f}", f"{pct}%"])

    # Auto width calculation
    def auto_col_widths(pdf, rows, min_width=20, max_width=70):
        num_cols = len(rows[0])
        widths = [min_width] * num_cols
        for row in rows:
            for i, cell in enumerate(row):
                w = pdf.get_string_width(str(cell)) + 4
                if w > widths[i]:
                    widths[i] = min(max_width, w)
        return widths

    col_widths = auto_col_widths(pdf, dept_rows)
    _paged_table_to_pdf(pdf, headers, dept_rows, col_widths, font_size=8)

    # -------------------- Yearly Summary --------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Yearly Summary", 0, 1)
    yearly_summary = summary_df.copy()
    yearly_summary["percent_change"] = yearly_summary.apply(
        lambda r: round((r["this_year"] - r["previous_year"]) / r["previous_year"] * 100, 2)
        if r["previous_year"] != 0 else ("∞" if r["this_year"] > 0 else 0),
        axis=1,
    )
    grand_total = yearly_summary[["previous_year", "this_year"]].sum()
    headers = ["Department", "Previous Year", "This Year", "% Change"]
    rows = [[r["department"], f"{r['previous_year']:,.2f}", f"{r['this_year']:,.2f}", f"{r['percent_change']}%"] for _, r in yearly_summary.iterrows()]
    rows.append([
        "GRAND TOTAL",
        f"{grand_total['previous_year']:,.2f}",
        f"{grand_total['this_year']:,.2f}",
        f"{round((grand_total['this_year'] - grand_total['previous_year']) / grand_total['previous_year'] * 100, 2) if grand_total['previous_year'] != 0 else ('∞' if grand_total['this_year'] > 0 else 0)}%"
    ])
    col_widths = [60, 50, 50, 30]
    _paged_table_to_pdf(pdf, headers, rows, col_widths, font_size=9)

    pdf.output(output_path)



# -------------------------------
# Routes (upload + URL)
# -------------------------------
@app.post("/generate-budget")
async def generate_budget(file: UploadFile = File(...), instructions: Optional[str] = None, adjustments: Optional[List[Adjustment]] = Body(None)):
    # we ignore 'instructions' free-text; use JSON adjustments only
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
            monthly_df = aggregate_monthly(df)

            # build adjustment list
            adjustments_list = adjustments or []

            monthly_adj, summary_df, warnings = apply_adjustments_to_monthly(monthly_df, adjustments_list)

            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(monthly_adj, summary_df, tmp_pdf_file, "", warnings)
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
            monthly_df = aggregate_monthly(df)

            adjustments_list = request.adjustments or []

            monthly_adj, summary_df, warnings = apply_adjustments_to_monthly(monthly_df, adjustments_list)

            tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
            generate_budget_pdf(monthly_adj, summary_df, tmp_pdf_file, "", warnings)
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
