import io
import os
import zipfile
import logging

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Ken French Data Library direct download URLs
_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)

# dowload zip from Ken French's site
def _fetch_french_csv(url) :

    logger.info("Fetching %s ...", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]           
        return zf.read(name).decode("utf-8", errors="replace")


def _parse_french_csv(raw, factor_cols) :
    
    lines = raw.splitlines()

    # Find the first line ~  a monthly date (6-digit YYYYMM)
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[:6].isdigit() and len(stripped.split(",")[0].strip()) == 6:
            start = i
            break

    if start is None:
        raise ValueError("Could not find start of data in French CSV")

    # Collect monthly rows until we hit a blank line 
    rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:          
            break
        parts = [p.strip() for p in stripped.split(",")]
        if not parts[0].isdigit():
            break
        rows.append(parts)

    # Build DataFrame; first column is YYYYMM date
    dates = []
    data_rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        parts = [p.strip() for p in stripped.split(",")]
        if not parts[0].isdigit():
            break
        # Parse date separately — never put a Timestamp into a str column
        dates.append(
            pd.to_datetime(parts[0], format="%Y%m") + pd.offsets.MonthEnd(0)
        )
        data_rows.append(parts[1: len(factor_cols) + 1])
    df = pd.DataFrame(data_rows, columns=factor_cols)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df.index = pd.DatetimeIndex(dates[: len(df)], name="date")
    df = df / 100
    return df


def download_factors(save_path: str = "data/processed/factors.csv") :
    """
    Download monthly Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA)
    and the Momentum factor (MOM) directly from the Ken French Data Library,
    then merge them into a single DataFrame.
    
    Returns
    pd.DataFrame
        Monthly factor returns indexed by end-of-month date, columns:
        Mkt-RF, SMB, HML, RMW, CMA, MOM, RF
    """
    logger.info("Starting factor download step")

    # 1. Fama-French 5 factors                                           
    ff5_raw = _fetch_french_csv(_FF5_URL)
    ff5     = _parse_french_csv(ff5_raw, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

    # 2. Momentum factor                                                 
    mom_raw = _fetch_french_csv(_MOM_URL)
    mom     = _parse_french_csv(mom_raw, ["MOM"])

   
    # 3. Merge and filter to 2017-2025                                    
    factor_df = ff5.join(mom, how="inner")
    factor_df = factor_df[
        (factor_df.index >= "2017-01-01") & (factor_df.index <= "2025-12-31")
    ]

    logger.info(
        "Factor data ready: %d months, columns: %s",
        len(factor_df),
        list(factor_df.columns),
    )

    # 4. Save                                                            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    factor_df.to_csv(save_path)
    logger.info("Factors saved to %s", save_path)

    return factor_df
