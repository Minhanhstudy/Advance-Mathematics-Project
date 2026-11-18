"""Shared helpers for the analysis scripts."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "buildings2021NL_repo.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

START_YEAR = 1950   # earlier years are sparse and most affected by demolition
END_YEAR = 2020     # 2021 is a partial year (data end in November 2021)


def logistic(t, K, r, t0):
    """Closed-form solution of dN/dt = r N (1 - N/K)."""
    return K / (1.0 + np.exp(-r * (t - t0)))


def load_annual_counts() -> pd.DataFrame:
    """Number of buildings built per year, and the cumulative stock N(t)."""
    df = pd.read_csv(RAW, usecols=["Constr_yea"])
    years = np.arange(df["Constr_yea"].min(), 2022)
    counts = df["Constr_yea"].value_counts().reindex(years, fill_value=0).sort_index()
    out = pd.DataFrame({"year": counts.index, "built": counts.values})
    out["cumulative"] = out["built"].cumsum()
    return out


def fit_cumulative(sub: pd.DataFrame):
    """Non-linear least squares of the logistic curve on the cumulative stock."""
    t = sub["year"].to_numpy(dtype=float)
    N = sub["cumulative"].to_numpy(dtype=float)
    p0 = [N.max() * 1.2, 0.08, t.mean()]
    popt, pcov = curve_fit(logistic, t, N, p0=p0, maxfev=50_000)
    perr = np.sqrt(np.diag(pcov))
    resid = N - logistic(t, *popt)
    rmse = float(np.sqrt(np.mean(resid**2)))
    r2 = 1 - np.sum(resid**2) / np.sum((N - N.mean()) ** 2)
    return popt, perr, resid, rmse, r2
