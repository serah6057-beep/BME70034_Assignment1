import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from linearmodels.panel import PanelOLS, FamaMacBeth
import warnings
warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BETA_COLS = ["beta_Mkt_RF", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA", "beta_MOM", "beta_CAT"]


def run_panel_regressions(panel_path: str = "data/processed/panel_betas.csv") -> dict:
    """
    Returns) dict : 각 방법의 회귀 결과 객체
    """

    # Load Data

    logger.info("Loading Panel Data")
    df = pd.read_csv(panel_path)
    df["date"]   = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(str)

    df = df.dropna(subset=BETA_COLS + ["excess_ret"])
    logger.info("데이터 shape: %s", df.shape)

    # (a) Pooled OLS
    
    logger.info("(a) Pooled OLS")
    X_pooled = add_constant(df[BETA_COLS])
    y        = df["excess_ret"]
    result_a = OLS(y, X_pooled).fit()
    logger.info("(a) finished")

    # linearmodels(PanelOLS, FamaMacBeth) MultiIndex 설정 (permno, date)

    df_panel = df.set_index(["permno", "date"])

    # (b) Fama-MacBeth + Newey-West SE (lags=6)
    
    logger.info("(b) Fama-MacBeth Regression")
    fm_model = FamaMacBeth(
        df_panel["excess_ret"],
        df_panel[BETA_COLS]
    )
    result_b = fm_model.fit(cov_type="kernel", bandwidth=6)
    logger.info("(b) finished")

    # 5. (c) Pooled OLS + date & stock fixed affects

    logger.info("(c) Pooled OLS + stock & date fixed affects")
    fe_model = PanelOLS(
        df_panel["excess_ret"],
        df_panel[BETA_COLS],
        entity_effects=True,   # stock fixed
        time_effects=True      # date fixed
    )
    result_c = fe_model.fit(cov_type="unadjusted")
    logger.info("(c) finished")

    # 6. (d) Pooled OLS + fixed + two-way clustered SE

    logger.info("(d) Pooled OLS + fixed + two-way clustered SE")
    result_d = fe_model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    logger.info("(d) finished")

    return {
        "pooled_ols":     result_a,
        "fama_macbeth":   result_b,
        "fe_ols":         result_c,
        "fe_ols_cluster": result_d,
    }


def make_regression_table(results) :
    
    labels = {
        "pooled_ols":     "(a) Pooled OLS",
        "fama_macbeth":   "(b) Fama-MacBeth",
        "fe_ols":         "(c) FE OLS",
        "fe_ols_cluster": "(d) FE OLS 2-way Cluster",
    }

    rows = []

    for key, res in results.items():
        col_name = labels[key]

        # 계수와 표준오차 추출
        # statsmodels(a)와 linearmodels(b,c,d)는 속성명이 조금 달라요
        if key == "pooled_ols":
            params = res.params
            pvalues = res.pvalues
            bse    = res.bse
            nobs   = int(res.nobs)
            r2     = res.rsquared
        else:
            params  = res.params
            pvalues = res.pvalues
            bse     = res.std_errors
            nobs    = int(res.nobs)
            r2      = res.rsquared

        # For each Beta,  make SE row
        for beta in BETA_COLS:
            if beta not in params.index:
                continue
            coef   = params[beta]
            se     = bse[beta]
            pval   = pvalues[beta]

            # 유의성 별표
            if pval < 0.01:
                star = "***"
            elif pval < 0.05:
                star = "**"
            elif pval < 0.10:
                star = "*"
            else:
                star = ""

            rows.append({
                "variable": beta,
                "stat":     "coef",
                col_name:   f"{coef:.4f}{star}"
            })
            rows.append({
                "variable": beta,
                "stat":     "se",
                col_name:   f"({se:.4f})"
            })

        # N과 R² 추가
        rows.append({"variable": "N",  "stat": "", col_name: f"{nobs:,}"})
        rows.append({"variable": "R²", "stat": "", col_name: f"{r2:.4f}"})

    
    table = pd.DataFrame(rows).groupby(["variable", "stat"]).first().reset_index()

    return table