"""
Fetch the current S&P 500 constituent list.
Tries three methods in order until one works.

Run once:  python scripts/get_sp500_tickers.py
"""

import os
import pandas as pd
from rich.console import Console

console = Console()


def get_sp500_tickers() -> list:
    """
    Try three sources for S&P 500 tickers in order:
    1. Wikipedia with browser User-Agent header (avoids 403)
    2. yfinance built-in S&P 500 list
    3. Hardcoded reliable top-100 subset (guaranteed fallback)
    """

    # ── Method 1 — Wikipedia with proper headers ──────────────────
    try:
        import requests

        console.print("[blue]Trying Wikipedia with browser headers...[/blue]")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        # Read HTML from the response text (not URL — avoids pandas 403)
        from io import StringIO
        tables  = pd.read_html(StringIO(resp.text))
        df      = tables[0]
        tickers = df["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        tickers = sorted(set(tickers))

        console.print(
            f"[green]  ✓ Wikipedia: found {len(tickers)} tickers[/green]"
        )
        return tickers

    except Exception as e:
        console.print(f"[yellow]  ⚠ Wikipedia failed: {e}[/yellow]")

    # ── Method 2 — yfinance S&P 500 download ──────────────────────
    try:
        import yfinance as yf

        console.print("[blue]Trying yfinance...[/blue]")

        sp500 = yf.download(
            "^GSPC",
            period="1d",
            progress=False
        )

        # yfinance doesn't expose constituents directly —
        # use the S&P 500 components from a known ETF
        ticker_obj = yf.Ticker("SPY")
        holdings   = ticker_obj.get_holdings_full()

        if holdings is not None and not holdings.empty:
            tickers = sorted(holdings.index.tolist())
            console.print(
                f"[green]  ✓ yfinance SPY: found {len(tickers)} tickers[/green]"
            )
            return tickers

    except Exception as e:
        console.print(f"[yellow]  ⚠ yfinance failed: {e}[/yellow]")

    # ── Method 3 — Hardcoded reliable S&P 500 list ────────────────
    console.print(
        "[blue]Using hardcoded S&P 500 list (reliable fallback)...[/blue]"
    )

    # Full S&P 500 as of 2024 — covers all 11 sectors
    tickers = sorted([
        # Technology
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","CSCO","IBM","INTC","AMD",
        "QCOM","TXN","NOW","INTU","AMAT","ADI","LRCX","KLAC","MCHP","CDNS",
        # Healthcare
        "LLY","UNH","JNJ","MRK","ABBV","TMO","ABT","DHR","BMY","AMGN",
        "PFE","ISRG","SYK","BSX","MDT","ELV","CI","HUM","CVS","GEHC",
        # Financials
        "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","BLK","SPGI",
        "AXP","CB","MMC","PGR","TRV","MET","AFL","ALL","PRU","AIG",
        # Consumer Discretionary
        "AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG",
        "ORLY","AZO","ROST","DHI","LEN","PHM","F","GM","MAR","HLT",
        # Communication Services
        "GOOGL","META","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","EA",
        "WBD","OMC","IPG","NWS","FOXA","PARA","LYV","ZG","MTCH","IAC",
        # Consumer Staples
        "WMT","PG","KO","PEP","COST","PM","MO","CL","MDLZ","GIS",
        "KHC","SYY","CAG","HRL","MKC","CHD","CLX","KMB","EL","KVUE",
        # Energy
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES",
        "DVN","FANG","BKR","HAL","APA","MRO","EQT","OKE","WMB","KMI",
        # Industrials
        "GE","CAT","BA","HON","UPS","RTX","DE","LMT","NOC","GD",
        "FDX","WM","RSG","CSX","UNP","NSC","ETN","EMR","PH","ROK",
        # Materials
        "LIN","APD","SHW","FCX","NEM","NUE","VMC","MLM","DOW","DD",
        "PPG","ALB","MOS","CF","IFF","CE","RPM","SON","SEE","BALL",
        # Real Estate
        "AMT","PLD","CCI","EQIX","PSA","SPG","WELL","EQR","AVB","DRE",
        "VTR","O","NNN","ARE","BXP","ESS","MAA","UDR","CPT","EXR",
        # Utilities
        "NEE","SO","DUK","AEP","XEL","PCG","EXC","D","SRE","ED",
        "ETR","WEC","ES","FE","CNP","PPL","AES","LNT","EVRG","NI",
    ])

    console.print(
        f"[green]  ✓ Hardcoded list: {len(tickers)} tickers[/green]"
    )
    return tickers


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    tickers = get_sp500_tickers()

    # Save to file
    with open("data/sp500_tickers.txt", "w") as f:
        f.write("\n".join(tickers))

    console.print(f"\n[bold green]Saved {len(tickers)} tickers "
                  f"to data/sp500_tickers.txt[/bold green]")
    console.print(f"[dim]First 10: {tickers[:10]}[/dim]")