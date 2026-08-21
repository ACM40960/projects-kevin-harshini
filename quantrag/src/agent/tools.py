"""
Phase 3 - Agent tools.

Two tools given to Claude in Node B:
  1. get_financial_metric  - exact structured numbers (yfinance)
  2. Filing evidence is provided directly as retrieved context,
     not as a callable tool (retrieval already happened in Node A)

Design note: we use yfinance's financial statements rather than
re-parsing SEC XBRL, since yfinance already provides clean structured
data and this project has extensive documented EDGAR API instability
(see Phase 1/2 handovers). This is a deliberate, defensible choice.
"""

import yfinance as yf
from rich.console import Console

console = Console()


def get_financial_metric(ticker: str, statement: str, year: int) -> dict:
    """
    Fetch a full financial statement row-set for one company/year.

    Args:
        ticker:    stock ticker, e.g. "NVDA"
        statement: "income" | "balance" | "cashflow"
        year:      fiscal year, e.g. 2023

    Returns:
        dict of {line_item_name: value} for that year, or
        {"error": "..."} if unavailable
    """
    try:
        t = yf.Ticker(ticker)
        # map Claude's simple "income"/"balance"/"cashflow" choice to
        # the actual yfinance property that returns that statement
        statement_map = {
            "income":   t.financials,
            "balance":  t.balance_sheet,
            "cashflow": t.cashflow,
        }
        df = statement_map.get(statement)

        if df is None or df.empty:
            return {"error": f"No {statement} statement data for {ticker}"}

        # yfinance columns are fiscal year-end Timestamps
        # e.g. a column might literally be Timestamp('2023-09-30'),
        # so we match on just the .year part of it
        matching_cols = [c for c in df.columns if c.year == year]
        if not matching_cols:
            available_years = sorted(set(c.year for c in df.columns))
            return {
                "error": f"No {year} data for {ticker}. "
                         f"Available years: {available_years}"
            }

        col = matching_cols[0]
        result = df[col].dropna().to_dict()
        # Convert numpy types to plain floats for JSON/tool-use compatibility
        return {str(k): float(v) for k, v in result.items()}

    except Exception as e:
        # never let a bad ticker/year crash the whole agent run - hand
        # back a clean error message Claude can read and react to instead
        return {"error": f"Failed to fetch {statement} for {ticker}: {str(e)[:150]}"}


# Tool schema for Claude tool-use 

FINANCIAL_DATA_TOOL_SCHEMA = {
    "name": "get_financial_metric",
    "description": (
        "Fetch exact structured financial statement data for a company/year "
        "(income statement, balance sheet, or cash flow). Use this whenever "
        "a calculation requires precise numbers (revenue, capex, total "
        "assets, etc.) rather than inferring figures from prose."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker, e.g. NVDA"},
            "statement": {
                "type": "string",
                "enum": ["income", "balance", "cashflow"],
                "description": "Which financial statement to fetch",
            },
            "year": {"type": "integer", "description": "Fiscal year, e.g. 2023"},
        },
        "required": ["ticker", "statement", "year"],
    },
}

VIEW_EXTRACTION_TOOL_SCHEMA = {
    "name": "extract_investment_view",
    "description": "Submit the final structured investment view for this stock.",
    "input_schema": {
        "type": "object",
        "properties": {
            "direction":  {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "magnitude":  {"type": "number", "description": "Expected return delta, e.g. 0.08 = +8%"},
            "confidence": {"type": "number", "description": "0.0 to 1.0"},
            "reasoning":  {"type": "string", "description": "1-2 sentence justification"},
            "citation":   {"type": "string", "description": "Source reference from the evidence"},
        },
        "required": ["direction", "magnitude", "confidence", "reasoning", "citation"],
    },
}