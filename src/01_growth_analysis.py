"""
01_growth_analysis.py
=====================
Analysis 1: How does the number of Dutch logistics buildings grow over time?

Model (dynamical system):  dN/dt = r * N * (1 - N / K)      (logistic equation)
    N(t)  cumulative number of logistics buildings built up to year t
    r     intrinsic growth rate (per year)
    K     carrying capacity (saturation level)
Equilibria: N = 0 (unstable, f'(0) = r > 0) and N = K (stable, f'(K) = -r < 0).
Closed-form solution: N(t) = K / (1 + exp(-r (t - t0))).

Two independent estimates are compared:
  (A) non-linear least squares of the closed-form curve on cumulative counts;
  (B) the "phase-line" regression dN_t = a*N_t - b*N_t^2  (so r = a, K = a/b),
      which fits the differential equation itself and is NOT affected by the
      inflated R^2 that any monotone cumulative series gives.

IMPORTANT LIMITATIONS (also stated in the README):
  * The data contain only buildings that still existed in 2021 (survivorship
    bias): old buildings that were demolished are missing, so early years are
    under-counted.
  * Logistics function was assigned from 2020 company data and projected
    backwards, so N(t) is the accumulation of *currently existing* logistics
    buildings, not the history of the logistics sector.
  * 2021 is a partial year (data end in November 2021) and is excluded.

Input : data/raw/buildings2021NL_repo.csv
Output: figures/fig1_growth_logistic.pdf
        results/growth_fit_summary.csv
        results/growth_sensitivity_start_year.csv
        data/processed/annual_counts.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DOCS, END_YEAR, FIGURES, PROCESSED, RESULTS, START_YEAR,  # noqa: E402
                    fit_cumulative, load_annual_counts, logistic)

SENSITIVITY_STARTS = [1900, 1930, 1950, 1960, 1970, 1980]


def fit_phase_line(sub: pd.DataFrame):
    """
    (B) Phase-line regression: dN_t = N_{t+1} - N_t = a*N_t - b*N_t^2.
    Ordinary least squares without intercept; r = a, K = a / b.
    Uses the number built in year t+1 as the increment observed at level N_t.
    """
    N = sub["cumulative"].to_numpy(dtype=float)[:-1]
    dN = sub["built"].to_numpy(dtype=float)[1:]
    X = np.column_stack([N, -(N**2)])
    coef, *_ = np.linalg.lstsq(X, dN, rcond=None)
    a, b = coef
    r_hat, K_hat = a, a / b
    pred = X @ coef
    r2 = 1 - np.sum((dN - pred) ** 2) / np.sum((dN - dN.mean()) ** 2)
    return r_hat, K_hat, r2, N, dN


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    annual = load_annual_counts()
    annual.to_csv(PROCESSED / "annual_counts.csv", index=False)

    sub = annual[(annual["year"] >= START_YEAR) & (annual["year"] <= END_YEAR)].copy()
    # Cumulative count is always from the first building (not from START_YEAR),
    # because N(t) is the total stock of buildings.

    # --- Fit (A) and (B) on the main window --------------------------------
    (K, r, t0), (K_se, r_se, t0_se), resid, rmse, r2_cum = fit_cumulative(sub)
    r_b, K_b, r2_phase, N_phase, dN_phase = fit_phase_line(sub)

    summary = pd.DataFrame(
        [
            {
                "method": "A: NLS on cumulative counts",
                "window": f"{START_YEAR}-{END_YEAR}",
                "K": K, "K_se": K_se, "r_per_year": r, "r_se": r_se,
                "inflection_year": t0, "R2": r2_cum, "RMSE_buildings": rmse,
            },
            {
                "method": "B: phase-line regression on annual increments",
                "window": f"{START_YEAR}-{END_YEAR}",
                "K": K_b, "K_se": np.nan, "r_per_year": r_b, "r_se": np.nan,
                "inflection_year": np.nan, "R2": r2_phase, "RMSE_buildings": np.nan,
            },
        ]
    )
    summary.to_csv(RESULTS / "growth_fit_summary.csv", index=False)
    print(summary.to_string(index=False))

    # --- Sensitivity to the start year -------------------------------------
    rows = []
    for s in SENSITIVITY_STARTS:
        sb = annual[(annual["year"] >= s) & (annual["year"] <= END_YEAR)]
        try:
            (Ks, rs, t0s), _, _, rm, r2c = fit_cumulative(sb)
            rb, Kb, r2p, _, _ = fit_phase_line(sb)
            rows.append({"start_year": s, "K_A": Ks, "r_A": rs, "t0_A": t0s,
                         "K_B": Kb, "r_B": rb, "R2_cumulative": r2c,
                         "R2_phase_line": r2p})
        except RuntimeError:
            rows.append({"start_year": s})
    sens = pd.DataFrame(rows)
    sens.to_csv(RESULTS / "growth_sensitivity_start_year.csv", index=False)
    print("\nSensitivity to the start year:")
    print(sens.round(3).to_string(index=False))

    # --- Figure -------------------------------------------------------------
    t = sub["year"].to_numpy(dtype=float)
    N_obs = sub["cumulative"].to_numpy(dtype=float)
    t_grid = np.linspace(START_YEAR, 2060, 500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) cumulative counts and fitted logistic curve (with projection)
    ax = axes[0]
    ax.plot(t, N_obs, "o", ms=3, color="#1f4e79", label="Observed (existing buildings)")
    ax.plot(t_grid, logistic(t_grid, K, r, t0), "-", color="#c0392b",
            label="Logistic fit")
    ax.axhline(K, ls="--", lw=0.8, color="grey")
    ax.text(2058, K * 0.94, f"K ≈ {K:,.0f}", fontsize=8, color="grey", ha="right")
    ax.axvline(t0, ls=":", lw=0.8, color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative number of buildings N(t)")
    ax.set_title("(a) Cumulative growth")
    ax.legend(fontsize=8, loc="upper left")

    # (b) phase line: dN vs N, with the parabola r N (1 - N/K)
    ax = axes[1]
    ax.plot(N_phase, dN_phase, "o", ms=3, color="#1f4e79", label="Observed (N_t, ΔN)")
    N_line = np.linspace(0, K_b * 1.02, 300)
    ax.plot(N_line, r_b * N_line * (1 - N_line / K_b), "-", color="#c0392b",
            label=f"r N (1 − N/K)  r={r_b:.3f}, K={K_b:,.0f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.plot([0], [0], "o", mfc="white", mec="black", label="N=0 (unstable)")
    ax.plot([K_b], [0], "o", color="black", label="N=K (stable)")
    ax.set_xlabel("Stock of buildings N")
    ax.set_ylabel("Buildings added per year, ΔN")
    ax.set_title("(b) Phase line of the logistic ODE")
    ax.legend(fontsize=7, loc="upper right")

    # (c) residuals of the cumulative fit
    ax = axes[2]
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(t, resid, "-o", ms=3, color="#1f4e79")
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (observed − fitted)")
    ax.set_title(f"(c) Residuals of fit A (RMSE ≈ {rmse:,.0f})")

    fig.suptitle(
        "Dutch logistics buildings: logistic growth model "
        f"({START_YEAR}–{END_YEAR}, buildings still existing in 2021)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_growth_logistic.pdf")
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / "fig1_growth_logistic.png", dpi=150)
    plt.close(fig)
    print("\nSaved figures/fig1_growth_logistic.pdf")


if __name__ == "__main__":
    main()
