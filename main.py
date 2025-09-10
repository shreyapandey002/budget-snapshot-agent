from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import tempfile
import os
from fpdf import FPDF
import uuid
from datetime import timedelta
from starlette.concurrency import run_in_threadpool


app = FastAPI(title="Budget Snapshot Agent")

# --------------------------------------
# Column synonyms (normalize Excel input)
# --------------------------------------
COLUMN_SYNONYMS = {
    "department": ["department", "company", "team", "function"],
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

    weekly_df = (
        df.groupby(["department", "week_start", "week_end"])["amount"]
        .sum()
        .reset_index()
    )
    weekly_df.rename(columns={"amount": "before_budget"}, inplace=True)
    return weekly_df


# ---------- Instruction Parser ----------
def parse_user_instructions(user_text: str):
    """
    Splits user instructions into three categories:
    - forecasts: high-level revenue/profit/growth targets
    - constraints: reductions (reduce, cut, decrease)
    - goals: increases (increase, boost, raise, expand)
    """

    parts = [p.strip() for p in user_text.replace("and", ",").split(",")]
    forecasts, constraints, goals = [], [], []

    for p in parts:
        p_lower = p.lower()

        # Forecasts (apply to overall company level)
        if any(keyword in p_lower for keyword in ["revenue", "profit", "growth"]):
            forecasts.append(p)

        # Explicit constraints (reductions)
        elif any(keyword in p_lower for keyword in ["reduce", "cut", "decrease"]):
            constraints.append(p)

        # Explicit goals (increases)
        elif any(keyword in p_lower for keyword in ["increase", "boost", "raise", "expand"]):
            goals.append(p)

        # Fallback: if contains a %, but no clear verb → treat as goal
        elif "%" in p_lower:
            goals.append(p)

    return forecasts, constraints, goals

# ---------- Apply Logic ----------
def apply_forecasts_constraints_goals(df: pd.DataFrame, forecasts, constraints, goals):
    df["after_budget"] = df["before_budget"].copy()
    
    # -----------------------
    # Step 1: Apply constraints (only if % is specified)
    # -----------------------
    constrained_depts = set()
    for c in constraints or []:
        c_lower = c.lower()
        for dept in df["department"].unique():
            if dept.lower() in c_lower:
                constrained_depts.add(dept)
                numbers = [int(s.strip('%')) for s in c.split() if s.strip('%').isdigit()]
                if numbers:
                    df.loc[df["department"] == dept, "after_budget"] *= (1 - numbers[0] / 100)

    # -----------------------
    # Step 2: Apply forecasts to non-constrained depts (only if % is specified)
    # -----------------------
    for f in forecasts or []:
        f_lower = f.lower()
        if "revenue" in f_lower or "profit" in f_lower or "growth" in f_lower:
            numbers = [int(s.strip('%')) for s in f.split() if s.strip('%').isdigit()]
            if numbers:
                multiplier = 1 + numbers[0] / 100
                df.loc[~df["department"].isin(constrained_depts), "after_budget"] *= multiplier

    # -----------------------
    # Step 3: Apply goals (only if % is specified)
    # -----------------------
    for g in goals or []:
        g_lower = g.lower()
        for dept in df["department"].unique():
            if dept.lower() in g_lower:
                numbers = [int(s.strip('%')) for s in g.split() if s.strip('%').isdigit()]
                if numbers:
                    if "increase" in g_lower:
                        df.loc[df["department"] == dept, "after_budget"] *= (1 + numbers[0] / 100)
                    elif "decrease" in g_lower or "reduce" in g_lower:
                        df.loc[df["department"] == dept, "after_budget"] *= (1 - numbers[0] / 100)

    # -----------------------
    # Step 4: Hybrid rebalance (only non-constrained departments)
    # -----------------------
    if constrained_depts:
        total_before = df["before_budget"].sum()
        constrained_before = df.loc[df["department"].isin(constrained_depts), "before_budget"].sum()
        constrained_after = df.loc[df["department"].isin(constrained_depts), "after_budget"].sum()

        remaining_after = df.loc[~df["department"].isin(constrained_depts), "after_budget"].sum()
        if remaining_after > 0:
            rebalancer = (total_before - constrained_after) / remaining_after
            df.loc[~df["department"].isin(constrained_depts), "after_budget"] *= rebalancer

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

    # Headers
    pdf.set_font("Arial", "B", 10)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, 1)
    pdf.ln()

    # Rows
    pdf.set_font("Arial", "", 10)
    for _, row in weekly_df.iterrows():
        week_range = f"{row['week_start'].strftime('%Y-%m-%d')} to {row['week_end'].strftime('%Y-%m-%d')}"
        percent_change = ((row['after_budget'] - row['before_budget']) / row['before_budget'] * 100) if row['before_budget'] else 0
        values = [
            row["department"],
            week_range,
            f"{row['before_budget']:.2f}",
            f"{row['after_budget']:.2f}",
            f"{percent_change:.2f}%",
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

    # Headers
    pdf.set_font("Arial", "B", 10)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, 1)
    pdf.ln()

    # Rows
    pdf.set_font("Arial", "", 10)
    for _, row in summary_df.iterrows():
        values = [
            row["department"],
            f"{row['before_budget']:.2f}",
            f"{row['after_budget']:.2f}",
            f"{row['percent_change']:.2f}%",
        ]
        for v, w in zip(values, col_widths):
            pdf.cell(w, 8, str(v), 1)
        pdf.ln()

    pdf.output(output_path)


# ---------- Routes ----------
@app.post("/generate-budget")
async def generate_budget(file: UploadFile, instructions: str = Form("")):
    suffix = os.path.splitext(file.filename)[1]
    if suffix.lower() not in [".xls", ".xlsx"]:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Only Excel files are supported."})

    # Save uploaded Excel temporarily
    fd, tmp_file_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    content = await file.read()
    with open(tmp_file_path, "wb") as f:
        f.write(content)

    # Define a sync worker
    def process_and_generate():
        df = pd.read_excel(tmp_file_path)
        df = normalize_columns(df)
        weekly_df = aggregate_weekly(df)
        forecasts, constraints, goals = parse_user_instructions(instructions)
        final_df = apply_forecasts_constraints_goals(weekly_df, forecasts, constraints, goals)
        summary_df = summarize_department_spending(final_df)

        tmp_pdf_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_budget_snapshot.pdf")
        generate_budget_pdf(final_df, summary_df, tmp_pdf_file)
        return tmp_pdf_file

    # Run heavy part in threadpool
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
