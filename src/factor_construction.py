import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Names of the seven factors used throughout
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "CAT"]
BETA_COLS   = [f"beta_{f.replace('-', '_')}" for f in FACTOR_COLS]

# Step A-2  CAT factor construction
def construct_cat_factor(stock_df, factor_df) :
    
    logger.info("CAT factor construction step")

    df = stock_df[["date", "permno", "ticker", "ret"]].copy()

    df["first_letter"] = df["ticker"].str[0].str.upper()
    c_stocks = df[df["first_letter"] == "C"]
    t_stocks = df[df["first_letter"] == "T"]

    c_ret = c_stocks.groupby("date")["ret"].mean().rename("ret_C")
    t_ret = t_stocks.groupby("date")["ret"].mean().rename("ret_T")

    cat = pd.concat([c_ret, t_ret], axis=1).dropna()

    # CAT = long C - short T (both legs equal-weighted)
    cat["CAT"] = cat["ret_C"] - cat["ret_T"]
    cat = cat[["CAT"]]

    logger.info(
        "CAT factor constructed: %d monthly observations (%s to %s)",
        len(cat),
        cat.index.min().date(),
        cat.index.max().date(),
    )
    return cat

# Step A-3  Rolling 7-factor time-series regressions
def _rolling_regression_for_stock(stock_panel, dates, factor_df, window = 24, min_obs = 10) :
    
    # Index stock returns by date for quick lookup
    ret_series = stock_panel.set_index("date")["excess_ret"]

    records = []
    for t in dates:
        window_end   = t - pd.offsets.MonthEnd(1)
        window_start = t - pd.offsets.MonthEnd(window) 

        y = ret_series.loc[
            (ret_series.index >= window_start) & (ret_series.index <= window_end)
        ]
        X = factor_df.loc[
            (factor_df.index >= window_start) & (factor_df.index <= window_end),
            FACTOR_COLS,
        ]

        # Align y and X on shared dates
        idx = y.index.intersection(X.index)
        y_  = y.loc[idx]
        X_  = X.loc[idx]

        # Skip if too few observations
        if len(idx) < min_obs:
            continue

        # OLS regression with intercept
        try:
            res = OLS(y_, add_constant(X_)).fit()
            betas = dict(zip(BETA_COLS, res.params[1:]))  # skip intercept
            betas["date"] = t
            records.append(betas)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("date")


def estimate_factor_loadings(stock_df, factor_df, window = 24, min_obs = 10) :
    """
    Returns
    -------
    pd.DataFrame
        Stock-month panel with columns:
        permno, date, ticker, excess_ret, beta_Mkt_RF, beta_SMB, beta_HML,
        beta_RMW, beta_CMA, beta_MOM, beta_CAT
    """
    logger.info("Starting rolling factor loading estimation")

    # Merge RF into stock panel to compute excess returns if needed
    if stock_df["excess_ret"].isna().all():
        rf = factor_df[["RF"]].rename(columns={"RF": "rf"})
        stock_df = stock_df.merge(rf, left_on="date", right_index=True, how="left")
        stock_df["excess_ret"] = stock_df["ret"] - stock_df["rf"].fillna(0)

    # Add CAT factor to factor_df for regressions
    # (factor_df must already contain the CAT column before calling this fn)
    assert "CAT" in factor_df.columns, (
        "factor_df must contain the 'CAT' column. "
        "Call construct_cat_factor() first and merge CAT in."
    )

    # Determine months for which we compute betas (2019-01 to 2025-12)
    all_dates = pd.DatetimeIndex(
        sorted(stock_df["date"].unique())
    )
    eval_dates = all_dates[all_dates >= "2019-01-01"]

    permnos = stock_df["permno"].unique()
    logger.info("Estimating betas for %d stocks across %d months", len(permnos), len(eval_dates))

    all_betas = []
    for i, permno in enumerate(permnos):
        if i % 500 == 0:
            logger.info("Processing stock %d / %d", i, len(permnos))

        sub = stock_df[stock_df["permno"] == permno][["date", "excess_ret"]]
        betas = _rolling_regression_for_stock(sub, eval_dates, factor_df, window, min_obs)
        if betas.empty:
            continue
        betas["permno"] = permno
        all_betas.append(betas.reset_index())

    if not all_betas:
        logger.warning("No betas estimated — check your stock and factor data.")
        return pd.DataFrame()

    beta_panel = pd.concat(all_betas, ignore_index=True)

    # Merge back ticker and excess_ret at month t
    meta = stock_df[["permno", "date", "ticker", "excess_ret"]].copy()
    panel = beta_panel.merge(meta, on=["permno", "date"], how="left")

    # Drop rows where excess_ret is missing
    panel = panel.dropna(subset=["excess_ret"] + BETA_COLS)

    logger.info(
        "Beta estimation complete: %d stock-month observations, "
        "%d unique stocks, %s to %s",
        len(panel),
        panel["permno"].nunique(),
        panel["date"].min().date(),
        panel["date"].max().date(),
    )

    return panel

# Convenience wrapper kept for backward compatibility with main.py stub
def construct_cat_factor_simple(stock_df) :
    
    logger.info("CAT factor construction step (stub)")
    return pd.DataFrame(columns=["CAT"])
