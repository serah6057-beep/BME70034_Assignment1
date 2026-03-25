import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Winsorize a single series at the given quantile bounds
def _winsorize_column(series, lower = 0.01, upper = 0.99) :
    
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)

# Winsorize columns cross-sectionally each month
def _winsorize_cross_sectional(df, date_col, columns, lower = 0.01, upper = 0.99) :

    logger.info("Winsorizing columns %s cross-sectionally", columns)
    df = df.copy()
    for col in columns:
        df[col] = df.groupby(date_col)[col].transform(
            lambda s: _winsorize_column(s, lower, upper)
        )
    return df

# Load CRSP monthly stock file (2017-01 to 2024-12)
def _load_crsp(path) :
    
    logger.info("Loading CRSP data from %s", path)
    df = pd.read_csv(path, low_memory=False)

    # Standardise column names
    df.columns = df.columns.str.lower().str.strip()

    # Rename to unified schema
    df = df.rename(columns={
        "permno": "permno",
        "date":   "date",
        "shrcd":  "shrcd",
        "exchcd": "exchcd",
        "ticker": "ticker",
        "prc":    "prc",
        "ret":    "ret",
        "shrout": "shrout",
    })

    df["source"] = "crsp"
    return df[["permno", "date", "shrcd", "exchcd", "ticker", "prc", "ret", "shrout", "source"]]

# Load Compustat Security Monthly file (2025-01 to 2025-12)
def _load_compustat(path) :

    logger.info("Loading Compustat data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # Rename to unified schema
    df = df.rename(columns={
        "tic":      "ticker",
        "datadate": "date",
        "gvkey":    "permno",  
        "exchg":    "exchcd",
        "tpci":     "shrcd",
        "prccm":    "prc",
        "trt1m":    "ret",
        "cshom":    "shrout",
    })

    # Compustat trt1m is in percentage — convert to decimal to match CRSP
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce") / 100

    # Compustat exchg codes -> map to CRSP-style exchcd
    # Compustat: 11=NYSE, 12=NYSE American (AMEX), 14=NASDAQ
    # CRSP:       1=NYSE,  2=AMEX,               3=NASDAQ
    exchg_map = {11: 1, 12: 2, 14: 3}
    df["exchcd"] = df["exchcd"].map(exchg_map)

    df["source"] = "compustat"
    return df[["permno", "date", "shrcd", "exchcd", "ticker", "prc", "ret", "shrout", "source"]]


def load_stock_data( crsp_path, compustat_path):
    
    logger.info("Starting stock data loading step")

    # 1. Load both data
    crsp       = _load_crsp(crsp_path)
    compustat  = _load_compustat(compustat_path)

    # 2. Merge two data
    df = pd.concat([crsp, compustat], ignore_index=True)
    logger.info("Combined raw shape: %s", df.shape)

    # 3. Parse dates and filter to 2017-01 through 2025-12
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df = df[(df["date"].dt.year >= 2017) & (df["date"].dt.year <= 2025)]

    # 4. Exchange filter: NYSE=1, AMEX=2, NASDAQ=3
    df = df[df["exchcd"].isin([1, 2, 3])]
    logger.info("After exchange filter: %d rows", len(df))

    # 5. Common share filter
    # CRSP: shrcd 10 or 11 (ordinary common shares)
    # Compustat: tpci == 0 (common stock)
    crsp_mask       = (df["source"] == "crsp")       & (df["shrcd"].isin([10, 11]))
    compustat_mask  = (df["source"] == "compustat")  & (df["shrcd"] == 0)
    df = df[crsp_mask | compustat_mask]
    logger.info("After share code filter: %d rows", len(df))

    # 6. Return filter: drop missing and CRSP sentinel values
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df = df.dropna(subset=["ret"])
    df = df[~df["ret"].isin([-66.0, -77.0, -88.0, -99.0, -999.0])]
    logger.info("After return filter: %d rows", len(df))

    # 7. Ticker filter
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"] != ""]

    # 8. Price filter
    df["prc"] = pd.to_numeric(df["prc"], errors="coerce").abs()
    df = df[df["prc"] > 0]

    # 9. Winsorize returns cross-sectionally at 1%/99% each month
    df = _winsorize_cross_sectional(df, date_col="date", columns=["ret"])

    # 10. Placeholder for excess return (filled after factor merge in main.py)
    df["excess_ret"] = np.nan

    # 11. Sort and reset index
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    logger.info(
        "Stock data loaded: %d stock-month observations, "
        "%d unique stocks, %s to %s",
        len(df),
        df["permno"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )

    return df
