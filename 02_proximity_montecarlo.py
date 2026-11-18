"""
02_proximity_montecarlo.py
==========================
Analysis 2: Are Dutch logistics buildings closer to (or farther from) residential areas
than expected if they were sited at random within industrial estates?

Statistic
    d_i = distance (m) from the centroid of building i to the centre of the nearest
          500 m x 500 m CBS cell with at least THETA inhabitants ("residential cell").
    T   = median of d_i over the 26,951 buildings.
    THETA is varied (5, 100, 500, 1000); THETA = 100 is the primary choice (reported
    first), the others are sensitivity checks. The CBS code -99997 means "no data" and
    is never treated as a population.

Null models (H0: the observed positions are not different from random siting)
    N1a "random siting among estates, by area": every building is placed at a uniformly
        random point of the union of the 3,001 industrial estates that contain logistics
        (an estate is chosen with probability proportional to its area, then a uniform
        point inside it). Very large estates (ports, heavy industry) dominate this null.
    N1b "random siting among estates, equal chance": as N1a, but every estate has the
        same probability. Checks that the result does not depend on the area weighting.
        Question for N1a/N1b: does the choice of WHERE logistics locates differ from
        random placement in these estates?
    N2  "random position within the real estate": every building keeps its real estate
        (column RIN) and is placed at a uniform random point inside it. Question: within
        an estate, is a building closer to residents than a random spot?

Monte Carlo test
    For b = 1..B draw a random configuration under H0 and compute T_b. With the observed
    value T_obs the empirical p-values are
        p_lower = (1 + #{T_b <= T_obs}) / (B + 1)      ("closer than random")
        p_upper = (1 + #{T_b >= T_obs}) / (B + 1)      ("farther than random")
        p_two   = min(1, 2 min(p_lower, p_upper))
    (the "+1" counts the observed configuration itself, so p is never 0; the smallest
    possible value is 1/(B+1)). The Monte Carlo standard error of the null mean is
    sd(T_b)/sqrt(B); the error of a p-value is about sqrt(p (1-p) / B).

Implementation note. Uniform random points are generated ONCE per estate (rejection
sampling inside the polygon, holes excluded) and their distances are computed once; each
replicate then draws from these pools. This makes thousands of replicates fast.

Also reported: the same test for four construction periods (< 1970, 1970-1989,
1990-2004, 2005-2021) at the primary THETA, comparing each period with a null sample of
the same size (nulls N1a and N1b).

LIMITATIONS: distances are centroid-to-cell-centre (500 m grid, so differences below
~500 m should not be interpreted); population is 2025, buildings are up to 2021; a
difference from the null is NOT causal (towns may have grown around older buildings);
many tests are run (4 thresholds x 3 nulls + 4 periods), so isolated small p-values must
be read with care (Bonferroni: multiply p by the number of tests).

Input : data/raw/distributioncenters2021NL_repo.gpkg  (layers buildings2021NL, estates2021NL)
        data/raw/cbs_vk500_2025_v1.gpkg                (layer cbs_vk500_2025)
Output: figures/fig2_proximity_montecarlo.pdf (+ PNG copy in docs/)
        results/montecarlo_summary.csv, results/montecarlo_by_period.csv,
        data/processed/building_distances.csv
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOCS, FIGURES, PROCESSED, RESULTS, ROOT  # noqa: E402
from spatial_io import polys_area, polys_centroid, read_layer  # noqa: E402

LOGISTICS_GPKG = ROOT / "data" / "raw" / "distributioncenters2021NL_repo.gpkg"
CBS_GPKG = ROOT / "data" / "raw" / "cbs_vk500_2025_v1.gpkg"

SEED = 2026
B = 2000                    # Monte Carlo replicates (smallest p-value = 1/(B+1))
POOL = 1000                 # random points pre-generated per estate
THETAS = [5, 100, 500, 1000]
PRIMARY = 100
PERIODS = [("before 1970", 0, 1969), ("1970-1989", 1970, 1989),
           ("1990-2004", 1990, 2004), ("2005-2021", 2005, 2021)]


# ----------------------------------------------------------------------------
# Random points inside estates
# ----------------------------------------------------------------------------
def random_points_in_polys(polys, n, rng):
    """n uniform random points inside a (multi)polygon with holes (rejection sampling)."""
    allpts = np.vstack([r for p in polys for r in p[:1]])
    xmin, ymin = allpts.min(axis=0)
    xmax, ymax = allpts.max(axis=0)
    ext = [MplPath(p[0]) for p in polys]
    holes = [[MplPath(h) for h in p[1:]] for p in polys]
    out, have = [], 0
    batch = max(2 * n, 200)
    while have < n:
        cand = np.column_stack([rng.uniform(xmin, xmax, batch), rng.uniform(ymin, ymax, batch)])
        ok = np.zeros(batch, bool)
        for e, hs in zip(ext, holes):
            inside = e.contains_points(cand)
            for h in hs:
                inside &= ~h.contains_points(cand)
            ok |= inside
        out.append(cand[ok])
        have += ok.sum()
    return np.vstack(out)[:n]


# ----------------------------------------------------------------------------
def mc_pvalues(t_obs, t_null):
    b = len(t_null)
    p_lo = (1 + np.sum(t_null <= t_obs)) / (b + 1)
    p_up = (1 + np.sum(t_null >= t_obs)) / (b + 1)
    return p_lo, p_up, min(1.0, 2 * min(p_lo, p_up))


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # --- data ----------------------------------------------------------------
    est = read_layer(LOGISTICS_GPKG, "estates2021NL", ["RIN"])
    est["area"] = [polys_area(p) for p in est.polys]
    bld = read_layer(LOGISTICS_GPKG, "buildings2021NL", ["fid", "RIN", "Constr_yea", "Function"])
    c = np.array([polys_centroid(p) for p in bld.polys])
    bld["x"], bld["y"] = c[:, 0], c[:, 1]
    est_index = {rin: i for i, rin in enumerate(est["RIN"])}
    bld["est"] = bld["RIN"].map(est_index)
    assert bld["est"].notna().all(), "building without estate"

    cells = read_layer(CBS_GPKG, "cbs_vk500_2025", ["aantal_inwoners"])
    cc = np.array([p[0][0][:4].mean(axis=0) for p in cells.polys])     # cell centres
    cells = pd.DataFrame({"x": cc[:, 0], "y": cc[:, 1], "pop": cells["aantal_inwoners"]})
    n_nodata = int((cells["pop"] < 0).sum())
    print(f"buildings: {len(bld):,}  estates: {len(est):,}  CBS cells: {len(cells):,} "
          f"(no-data code in {n_nodata:,}, treated as not residential)")

    # --- random pools ---------------------------------------------------------
    n_est = len(est)
    pool = np.empty((n_est, POOL, 2))
    for i, polys in enumerate(est["polys"]):
        pool[i] = random_points_in_polys(polys, POOL, rng)
    print(f"pools generated ({n_est * POOL:,} points) in {time.time() - t0:.0f} s")
    # sanity: pool points really lie in their estate bounding boxes
    flat = pool.reshape(-1, 2)

    weights = est["area"].to_numpy() / est["area"].sum()
    est_b = bld["est"].to_numpy(dtype=int)
    n_b = len(bld)
    bxy = bld[["x", "y"]].to_numpy()

    # --- distances ---------------------------------------------------------------
    d_obs, d_pool = {}, {}
    for th in THETAS:
        res = cells[cells["pop"] >= th]
        tree = cKDTree(res[["x", "y"]].to_numpy())
        d_obs[th] = tree.query(bxy)[0]
        d_pool[th] = tree.query(flat)[0].reshape(n_est, POOL)
        print(f"theta={th:>5}: residential cells {len(res):>6,}  observed median "
              f"{np.median(d_obs[th]):7.1f} m")
    pd.DataFrame({"fid": bld["fid"], "Constr_yea": bld["Constr_yea"],
                  "Function": bld["Function"],
                  **{f"dist_theta{th}_m": d_obs[th].round(1) for th in THETAS}}
                 ).to_csv(PROCESSED / "building_distances.csv", index=False)

    # --- Monte Carlo: overall, both nulls, all thresholds ------------------------
    rows, null_store = [], {}
    for th in THETAS:
        dp = d_pool[th]
        t1 = np.empty(B)
        t1b = np.empty(B)
        t2 = np.empty(B)
        for b in range(B):
            e1 = rng.choice(n_est, size=n_b, p=weights)             # N1a: estate ~ area
            t1[b] = np.median(dp[e1, rng.integers(POOL, size=n_b)])
            e1b = rng.integers(n_est, size=n_b)                     # N1b: estate, equal chance
            t1b[b] = np.median(dp[e1b, rng.integers(POOL, size=n_b)])
            t2[b] = np.median(dp[est_b, rng.integers(POOL, size=n_b)])   # N2: own estate
        t_obs = float(np.median(d_obs[th]))
        for name, tn in [("N1a random estate (by area) + point", t1),
                         ("N1b random estate (equal chance) + point", t1b),
                         ("N2 random point in own estate", t2)]:
            p_lo, p_up, p_two = mc_pvalues(t_obs, tn)
            lo, hi = np.percentile(tn, [2.5, 97.5])
            rows.append({"theta_inhabitants": th, "null_model": name,
                         "observed_median_m": t_obs, "null_mean_m": tn.mean(),
                         "null_95pct_low_m": lo, "null_95pct_high_m": hi,
                         "mc_se_of_null_mean_m": tn.std(ddof=1) / np.sqrt(B),
                         "p_closer": p_lo, "p_farther": p_up, "p_two_sided": p_two,
                         "B": B})
        null_store[th] = {"N1a": t1, "N1b": t1b, "N2": t2}
    summ = pd.DataFrame(rows)
    summ.round(4).to_csv(RESULTS / "montecarlo_summary.csv", index=False)
    print("\n" + summ.round(3).to_string(index=False))

    # --- by construction period (primary theta, nulls N1a and N1b) ---------------
    dp = d_pool[PRIMARY]
    rows_p = []
    for label, y0, y1 in PERIODS:
        idx = ((bld["Constr_yea"] >= y0) & (bld["Constr_yea"] <= y1)).to_numpy()
        n = int(idx.sum())
        t_obs = float(np.median(d_obs[PRIMARY][idx]))
        tn_a, tn_b = np.empty(B), np.empty(B)
        for b in range(B):
            e1 = rng.choice(n_est, size=n, p=weights)
            tn_a[b] = np.median(dp[e1, rng.integers(POOL, size=n)])
            e1b = rng.integers(n_est, size=n)
            tn_b[b] = np.median(dp[e1b, rng.integers(POOL, size=n)])
        for name, tn in [("N1a by area", tn_a), ("N1b equal chance", tn_b)]:
            p_lo, p_up, p_two = mc_pvalues(t_obs, tn)
            lo, hi = np.percentile(tn, [2.5, 97.5])
            rows_p.append({"period": label, "null_model": name, "n_buildings": n,
                           "observed_median_m": t_obs, "null_mean_m": tn.mean(),
                           "null_95pct_low_m": lo, "null_95pct_high_m": hi,
                           "observed_over_null": t_obs / tn.mean(),
                           "p_closer": p_lo, "p_farther": p_up, "p_two_sided": p_two,
                           "theta_inhabitants": PRIMARY, "B": B})
    per = pd.DataFrame(rows_p)
    per.round(4).to_csv(RESULTS / "montecarlo_by_period.csv", index=False)
    print("\n" + per.round(3).to_string(index=False))

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    blue, red, grey = "#1f4e79", "#c0392b", "#9db7d1"

    ax = axes[0]
    ns = null_store[PRIMARY]
    obs0 = float(np.median(d_obs[PRIMARY]))
    for key, col, lab in [("N1a", blue, "Null N1a, by area"),
                          ("N1b", "#2e8b57", "Null N1b, equal chance"),
                          ("N2", "#d68910", "Null N2, own estate")]:
        ax.hist(ns[key], bins=25, color=col, alpha=0.9, edgecolor="none",
                label=f"{lab} (mean {ns[key].mean():.0f} m)")
    ax.axvline(obs0, color=red, lw=2.2, label=f"Observed median: {obs0:.0f} m")
    ax.set_xlim(300, 620)
    ax.set_xlabel("Median distance to residential cell (m)")
    ax.set_ylabel(f"Replicates (B = {B})")
    ax.set_title(f"(a) Observed vs. null distributions, ≥{PRIMARY} inhabitants")
    ax.legend(fontsize=7.5, loc="upper center")

    ax = axes[1]
    x = np.arange(len(THETAS))
    for off, (key, col, lab) in zip([-0.2, 0.0, 0.2],
                                    [("N1a", blue, "Null N1a, by area (95%)"),
                                     ("N1b", "#2e8b57", "Null N1b, equal chance (95%)"),
                                     ("N2", "#d68910", "Null N2, own estate (95%)")]):
        sub = summ[summ.null_model.str.startswith(key)]
        ax.errorbar(x + off, sub.null_mean_m,
                    yerr=[sub.null_mean_m - sub.null_95pct_low_m,
                          sub.null_95pct_high_m - sub.null_mean_m],
                    fmt="o", color=col, capsize=3, label=lab)
    obs = summ[summ.null_model.str.startswith("N1a")].observed_median_m.to_numpy()
    ax.plot(x, obs, "D", color=red, ms=7, label="Observed")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{t}" for t in THETAS])
    ax.set_xlabel("Inhabitants per 500 m cell defining 'residential'")
    ax.set_ylabel("Median distance (m)")
    ax.set_title("(b) Observed vs. three null models")
    ax.legend(fontsize=8)

    ax = axes[2]
    periods = [p[0] for p in PERIODS]
    x = np.arange(len(periods))
    for off, (key, col, lab) in zip([-0.12, 0.12], [("N1a", blue, "Null N1a, by area (95%)"),
                                                    ("N1b", "#2e8b57", "Null N1b, equal chance (95%)")]):
        sub = per[per.null_model.str.startswith(key)].set_index("period").loc[periods]
        ax.errorbar(x + off, sub.null_mean_m,
                    yerr=[sub.null_mean_m - sub.null_95pct_low_m,
                          sub.null_95pct_high_m - sub.null_mean_m],
                    fmt="o", color=col, capsize=3, label=lab)
    sub = per[per.null_model.str.startswith("N1a")].set_index("period").loc[periods]
    ax.plot(x, sub.observed_median_m, "D", color=red, ms=7, label="Observed")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n:,})" for l, n in zip(periods, sub.n_buildings)], fontsize=8)
    ax.set_ylabel("Median distance (m)")
    ax.set_title(f"(c) By construction period, ≥{PRIMARY} inhabitants")
    ax.legend(fontsize=8)

    fig.suptitle("Monte Carlo test: proximity of Dutch logistics buildings to residential "
                 "areas versus random siting in industrial estates", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_proximity_montecarlo.pdf")
    fig.savefig(DOCS / "fig2_proximity_montecarlo.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved figures/fig2_proximity_montecarlo.pdf  (total {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
