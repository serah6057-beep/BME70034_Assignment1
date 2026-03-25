import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 7 factors 6 factors

SEVEN_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "CAT"]
SIX_FACTORS   = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]

# 1. Construct 25 portfolios

def make_25_portfolios(panel_df, other_factor = "beta_HML") :
   
    logger.info("Construct 25 portfolios (CAT x %s)", other_factor)

    port_rets = []

    for date, group in panel_df.groupby("date"):
        cat_q   = pd.qcut(group["beta_CAT"],   5, labels=[1,2,3,4,5], duplicates="drop")
        other_q = pd.qcut(group[other_factor], 5, labels=[1,2,3,4,5], duplicates="drop")

        group = group.copy()
        group["cat_q"]   = cat_q
        group["other_q"] = other_q
        group = group.dropna(subset=["cat_q", "other_q"])

        # equal-weighted monthly excess returns
        row = {"date": date}
        for i in range(1, 6):
            for j in range(1, 6):
                mask = (group["cat_q"] == i) & (group["other_q"] == j)
                ret  = group.loc[mask, "excess_ret"].mean()
                row[f"CAT_q{i}_{other_factor}_q{j}"] = ret
        port_rets.append(row)

    port_df = pd.DataFrame(port_rets).set_index("date")
    port_df.index = pd.to_datetime(port_df.index)
    logger.info("Finished: shape %s", port_df.shape)
    return port_df

# 2. GRS Stat Function

def calc_grs(alphas, resid, factor_means, factor_cov, T, N, K) :
    
    from scipy import stats

    # covariance matrix of redual: Sigma (N x N)
    Sigma = np.cov(resid.T)   # resid.T shape: (N, T)

    # factor-sharp ratio: 1 + mu' * Omega^-1 * mu
    Omega_inv = np.linalg.inv(factor_cov)
    sh2_f = float(factor_means @ Omega_inv @ factor_means)

    # GRS statistic
    Sigma_inv = np.linalg.inv(Sigma)
    grs_num   = float(alphas @ Sigma_inv @ alphas)
    grs_stat  = ((T - N - K) / N) * grs_num / (1 + sh2_f)

    # F 분포 p값: F(N, T-N-K)
    p_value = 1 - stats.f.cdf(grs_stat, N, T - N - K)

    return grs_stat, p_value

# 3. Time-Series Regression + GRS Test

def run_grs(port_df, factor_df, factors, label) :
    
    logger.info("%s GRS Test", label)

    factor_sub = factor_df[factors].dropna()
    common_dates = port_df.index.intersection(factor_sub.index)
    port_sub   = port_df.loc[common_dates].dropna(axis=1, how="all")
    factor_sub = factor_sub.loc[common_dates]

    T = len(common_dates)
    N = port_sub.shape[1]
    K = len(factors)

    # Regression to each portfolio → get alpha
    alphas  = []
    resids  = []

    for col in port_sub.columns:
        y = port_sub[col]
        X = add_constant(factor_sub)
        valid = y.notna()
        res   = OLS(y[valid], X[valid]).fit()
        alphas.append(res.params["const"])
        resids.append(res.resid.values)

    alphas_arr = np.array(alphas)           # (N,)
    resid_mat  = np.column_stack(resids)    # (T, N)

    # Calculate GRS Statistic
    f_means = factor_sub.mean().values
    f_cov   = factor_sub.cov().values
    grs_stat, p_value = calc_grs(alphas_arr, resid_mat, f_means, f_cov, T, N, K)

    avg_abs_alpha = np.mean(np.abs(alphas_arr))

    logger.info("%s → GRS=%.4f, p=%.4f, avg|α|=%.6f", label, grs_stat, p_value, avg_abs_alpha)

    # 알파를 5×5 행렬로 재구성
    port_names  = port_sub.columns.tolist()
    alpha_grid  = np.array(alphas_arr).reshape(5, 5)
    alphas_df   = pd.DataFrame(
        alpha_grid,
        index  =[f"CAT Q{i}" for i in range(1, 6)],
        columns=[f"Other Q{j}" for j in range(1, 6)]
    )

    return {
        "label":         label,
        "grs_stat":      grs_stat,
        "p_value":       p_value,
        "avg_abs_alpha": avg_abs_alpha,
        "alphas_df":     alphas_df,
    }

# 4. Heatmap

def plot_heatmap(alphas_df: pd.DataFrame, title: str, save_path: str):
   
    fig, ax = plt.subplots(figsize=(8, 6))

    vmax = np.abs(alphas_df.values).max()
    vmin = -vmax

    im = ax.imshow(
        alphas_df.values,
        cmap="RdBu_r",     
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )

    # alpha => each cell
    for i in range(5):
        for j in range(5):
            val = alphas_df.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=9, color="black")

    # label
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(alphas_df.columns)
    ax.set_yticklabels(alphas_df.index)
    ax.set_xlabel("Other Factor Quintile")
    ax.set_ylabel("CAT Beta Quintile")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Alpha")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Save Heatmap: %s", save_path)

# 5. 전체 Task C 실행 함수

def run_task_c(
    panel_path = "data/processed/panel_betas.csv",
    factor_path = "data/processed/factors_with_cat.csv",
    other_factor: str = "beta_HML") :
    
    # Data Load
    panel_df = pd.read_csv(panel_path)
    panel_df["date"] = pd.to_datetime(panel_df["date"])

    factor_df = pd.read_csv(factor_path, index_col=0)
    factor_df.index = pd.to_datetime(factor_df.index)

    # Construct 25 portfolios
    port_df = make_25_portfolios(panel_df, other_factor=other_factor)

    # 7-Factors GRS
    result_7 = run_grs(port_df, factor_df, SEVEN_FACTORS, "7-Factor")
    plot_heatmap(
        result_7["alphas_df"],
        title=f"7-Factor Alphas (CAT x {other_factor})",
        save_path="results/heatmap_7factor.png"
    )

    # 6-Factors GRS (exceptCAT 제외)
    result_6 = run_grs(port_df, factor_df, SIX_FACTORS, "6-Factor")
    plot_heatmap(
        result_6["alphas_df"],
        title=f"6-Factor Alphas (CAT x {other_factor})",
        save_path="results/heatmap_6factor.png"
    )

    # 결과 요약 저장
    summary = pd.DataFrame([
        {
            "Model":         result_7["label"],
            "GRS F-stat":    round(result_7["grs_stat"], 4),
            "p-value":       round(result_7["p_value"], 4),
            "Avg |Alpha|":   round(result_7["avg_abs_alpha"], 6),
        },
        {
            "Model":         result_6["label"],
            "GRS F-stat":    round(result_6["grs_stat"], 4),
            "p-value":       round(result_6["p_value"], 4),
            "Avg |Alpha|":   round(result_6["avg_abs_alpha"], 6),
        },
    ])
    summary.to_csv("results/grs_summary.csv", index=False)
    logger.info("\n%s", summary.to_string())

    return {"7factor": result_7, "6factor": result_6}