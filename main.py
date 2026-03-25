import os
import logging
import pandas as pd

from src.download_factors import download_factors
from src.data_processing import load_stock_data
from src.factor_construction import construct_cat_factor, estimate_factor_loadings
from src.panel_regression import run_panel_regressions, make_regression_table
from src.grs_test import run_task_c

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/assignment1.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def main():

    # Task A. Data Construction

    logger.info("Assignment 1 started")

    # Task A-1  Download six Fama-French + Momentum factors
    factor_df = download_factors(save_path="data/processed/factors.csv")
    logger.info("Factor columns: %s", list(factor_df.columns))

    # Task A-2  Load stock data and construct the CAT factor
    stock_df = load_stock_data(crsp_path="data/raw/stocks_crsp.csv",
    compustat_path="data/raw/stocks_compustat.csv")

    cat_df = construct_cat_factor(stock_df, factor_df)

    # Merge CAT into factor_df so rolling regressions can use all 7 factors
    factor_df = factor_df.join(cat_df, how="left")

    # Persist combined factor data (7 factors + RF)
    factor_df.to_csv("data/processed/factors_with_cat.csv")
    logger.info("Factors with CAT saved.")

    # Summary statistics for the CAT factor (required output)
    cat_summary = pd.DataFrame({
        "mean":   [cat_df["CAT"].mean()],
        "std":    [cat_df["CAT"].std()],
    })
    factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    corr_with_cat = factor_df[factor_names].corrwith(factor_df["CAT"]).rename("corr_with_CAT")
    logger.info("\nCAT factor summary:\n%s", cat_summary.to_string())
    logger.info("\nCorrelations with other factors:\n%s", corr_with_cat.to_string())
    cat_summary.to_csv("results/cat_summary.csv", index=False)
    corr_with_cat.to_frame().to_csv("results/cat_correlations.csv")

    # Task A-3 & A-4  Rolling 7-factor regressions -> stock-month panel
    panel_df = estimate_factor_loadings(
        stock_df=stock_df,
        factor_df=factor_df,
        window=24,
        min_obs=10,
    )

    # Save the panel for Part B (cross-sectional regressions)
    panel_df.to_csv("data/processed/panel_betas.csv", index=False)
    logger.info("Panel betas saved to data/processed/panel_betas.csv")

    print("\n=== Task A complete ===")
    print(f"Factor data shape    : {factor_df.shape}")
    print(f"Stock panel shape    : {stock_df.shape}")
    print(f"Beta panel shape     : {panel_df.shape}")
    print(f"Date range (betas)   : {panel_df['date'].min().date()} - {panel_df['date'].max().date()}")
    # ------------------------------------------------------------------------------------------------------
    # Task B. Regressions

    results = run_panel_regressions(panel_path="data/processed/panel_betas.csv")
    table   = make_regression_table(results)
    table.to_csv("results/regression_table.csv", index=False)
    logger.info("Result of regression is saved: results/regression_table.csv")
    print(table.to_string())

    run_task_c(
    panel_path="data/processed/panel_betas.csv",
    factor_path="data/processed/factors_with_cat.csv",
    other_factor="beta_HML"
)


if __name__ == "__main__":
    main()
