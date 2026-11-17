# Evaluating the Proximity of Logistics Facilities to Residential Neighbourhoods in the Netherlands Through Dynamical Systems and Monte Carlo Methods

Project for *Advanced Mathematics for Business and Technology* (VUB). Individual project.
**Website:** _TODO: add the URL of the published site (GitHub Pages, folder `docs/`)_

*Terminology.* "Logistics facilities" are the logistics buildings larger than 500 m² of the
dataset of Nefs (2022). "Residential neighbourhoods" are populated 500 m × 500 m grid cells
of Statistics Netherlands (CBS), not the official Dutch *buurt* units. The dynamical-systems
part models the **growth** of the stock of buildings; the Monte Carlo part tests their
**proximity** to residents.

## 1. Research questions

**Main question.** How close are logistics buildings in the Netherlands to residential areas,
and are they closer or farther than if they were sited at random within industrial estates?

**Secondary questions.**
1. Does the stock of logistics buildings follow a logistic growth law (a dynamical system), and
   what are its growth rate `r`, saturation level `K` and equilibria?
2. Does the distance to residential areas differ between buildings of different construction
   periods?

**Motivation.** I want to connect this project to my research paper, which studies how developers, logistics operators and public
authorities integrate urban logistics facilities into mixed-use neighbourhoods in Brussels.
*Proximity* between logistics and residents is a central tension there. Open building-level data
for Brussels are not available, so this project measures proximity in Netherlands, a neighbouring country
using open data, as a demonstration of the method. 

## 2. Data sources

| Data | Source | Licence |
|---|---|---|
| 26,951 logistics buildings > 500 m² (construction year, function, footprint) and the 3,001 industrial estates that contain them | Nefs, M. (2022). *Dutch Distribution Centres 2021 Geodata*. 4TU.ResearchData. https://doi.org/10.4121/19361018 | CC BY-SA 4.0 |
| Inhabitants per 500 m × 500 m cell (2025) | Statistics Netherlands (CBS), `cbs_vk500_2025_v1.gpkg` | to be confirmed (see `data/README.md`) |

Download instructions: [`data/README.md`](data/README.md).

## 3. Mathematics used

### 3.1 Dynamical systems (logistic growth)
Let `N(t)` be the number of buildings built up to year `t`. The model is the autonomous ODE

`dN/dt = f(N) = r N (1 − N/K)`

* **Equilibria:** `f(N*) = 0` gives `N* = 0` and `N* = K`.
* **Stability by linearisation:** near `N*`, `du/dt ≈ f'(N*) u` with `u = N − N*`, so
  `u(t) = u0 e^{f'(N*) t}`. Since `f'(N) = r(1 − 2N/K)`: `f'(0) = r > 0` (unstable) and
  `f'(K) = −r < 0` (stable). In one dimension `f'(N*)` is the single eigenvalue of the Jacobian.
* **Closed form (separation of variables):** `N(t) = K / (1 + e^{−r(t − t0)})`, with maximal
  growth `rK/4` at `N = K/2` (the inflection point `t0`).
* **Discrete version used for estimation:** `ΔN_t = a N_t − b N_t²`, so `r = a`, `K = a/b`.

### 3.2 Monte Carlo simulation
For a statistic `T` (here the median distance) and a null model H0, simulate `B` random
configurations, compute `T_1, …, T_B`, and compare with the observed `T_obs`:

`p_closer = (1 + #{T_b ≤ T_obs}) / (B + 1)`,  `p_farther = (1 + #{T_b ≥ T_obs}) / (B + 1)`,
`p_two-sided = min(1, 2 min(p_closer, p_farther))`.

The "+1" counts the observed configuration itself, so `p` is never 0 (smallest value `1/(B+1)`).
By the law of large numbers the simulated mean converges to its expectation; by the central
limit theorem its error shrinks like `σ/√B` (halving the error needs four times more replicates).

## 4. Analyses and results

### Analysis 1: logistic growth (`src/01_growth_analysis.py`, figure 1)
Window 1950–2020; two independent estimates:
**A** non-linear least squares of the closed-form curve on the cumulative counts;
**B** regression of the yearly increments on the stock (`ΔN = aN − bN²`).

| | K (buildings) | r (per year) | Inflection year | R² |
|---|---|---|---|---|
| A (cumulative) | 31,085 ± 383 | 0.084 ± 0.0014 | 1997 | 0.998 |
| B (phase line) | 29,966 | 0.088 | n/a | 0.69 |

The estimates are stable when the start year of the window changes
(`results/growth_sensitivity_start_year.csv`). **Reading it carefully:** the R² of A is inflated
because any monotone cumulative series is smooth, so B (R² ≈ 0.69) is the more honest goodness of
fit; the residuals oscillate systematically (construction booms and slowdowns), so the logistic
law is only a rough description; and `K` is only 12–16 % above the 2020 stock, a saturation level
for the stock of *currently existing* buildings, not a forecast.

### Analysis 2: Monte Carlo test of proximity (`src/02_proximity_montecarlo.py`, figure 2)
For each building: distance `d_i` from its centroid to the centre of the nearest cell with at
least `θ` inhabitants; statistic `T` = median of `d_i`. `θ = 100` is the primary choice; 5, 500
and 1000 are sensitivity checks. Three null models, `B = 2000` replicates each, seed 2026:

* **N1a:** random point in the union of the 3,001 estates (estate chosen by area);
* **N1b:** as N1a but every estate equally likely (checks the area weighting);
* **N2:** every building keeps its real estate and is placed at a random point inside it.

Primary result (`θ = 100`, median distance in metres, 95 % interval of the null in brackets):

| Observed | N1a (by area) | N1b (equal chance) | N2 (own estate) |
|---|---|---|---|
| **418** | 584 [575, 593] → closer, p ≤ 0.001 | 334 [330, 338] → **farther**, p ≤ 0.001 | 426 [423, 430] → closer by 2 % |

All p-values hit the floor `2/(B+1)` of the simulation; they are reported as "≤ 0.001", never
as exactly 0. Full table for all thresholds: `results/montecarlo_summary.csv`.

**What is robust and what is not.**
1. **The answer to "closer than random?" depends on the null model.** Against area-weighted
   estates (N1a) the buildings are closer; against equally likely estates (N1b) they are
   farther for `θ ≤ 500` and slightly closer for `θ = 1000`. No single "yes/no" is justified.
2. **Position within an estate matters little.** Under N2 the buildings are only 1–2 % closer
   than random spots in their own estate (about 8 m at `θ = 100`), far below the 500 m resolution
   of the grid. Proximity is mainly determined by *which* estate hosts logistics.
3. **Newer buildings are farther from today's residential cells.** Observed median distance
   (`θ = 100`): 286 m (before 1970), 376 m (1970–1989), 439 m (1990–2004), 512 m (2005–2021).
   The ratio observed/null rises with the period under both N1a (0.49 → 0.88) and N1b
   (0.86 → 1.53), so the direction of the trend does not depend on the null (`results/montecarlo_by_period.csv`).
   This is a **description, not a causal claim** (see limitations).

## 5. Limitations

* **Survivorship bias.** Only buildings still standing in 2021 are included; demolished buildings
  are missing, so early years are under-counted.
* **Backward reconstruction.** Logistics function was assigned from 2020 company data and projected
  backwards (see the dataset README), so `N(t)` is the accumulation of *currently existing*
  logistics buildings, not the history of the logistics sector. In my own count (1950–2020), 16 % of the
  buildings are dated to a year ending in 0 (10 % would be expected), which suggests that some
  construction years are rounded or estimated; this adds noise to the yearly counts.
* **Year 2021** is incomplete (data end in November 2021) and is excluded from Analysis 1.
* **Population grid.** 500 m resolution, rounded values, population of 2025 against buildings up to
  2021; the code `-99997` means "no data" and is never used as a population. Distances are
  centroid-to-cell-centre, so differences below about 500 m should not be interpreted.
* **Results depend on the definition of "residential"** (`θ`) and on the null model (see above).
* **A trend of distance with construction year is not causal.** A building constructed in 1960 may
  be close to residents because the town grew around it later.
* **Many tests** are reported (4 thresholds × 3 nulls + 4 periods × 2 nulls); read isolated small
  p-values with care (Bonferroni: multiply by the number of tests).
* **Scope** the whole sector, not a single company. No statement is made about
  noise, traffic or any effect on residents.

## 6. Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md                  how to obtain the raw data, licences
│   ├── raw/                       original downloads (large files are git-ignored)
│   └── processed/                 files produced by the scripts
├── src/
│   ├── common.py                  shared helpers
│   ├── spatial_io.py              minimal GeoPackage reader (no geopandas needed)
│   ├── 01_growth_analysis.py      analysis 1 (+ figure 1)
│   └── 02_proximity_montecarlo.py analysis 2 (+ figure 2)
├── figures/
│   ├── fig1_growth_logistic.pdf
│   └── fig2_proximity_montecarlo.pdf
├── results/                       CSV tables behind every number above
└── docs/                          website (index.html + PNG copies of the figures)
```

## 7. How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the data in data/raw/ as described in data/README.md
python src/01_growth_analysis.py        # a few seconds
python src/02_proximity_montecarlo.py   # about 75 s for B = 2000
```
Settings (`B`, `POOL`, `THETAS`, `SEED`) are at the top of `src/02_proximity_montecarlo.py`.

## 8. Use of AI tools

AI assistance (Claude) was used for brainstorming, data exploration, drafting code and text. I
checked the methods and results myself and can explain each step.

## 9. Licence and citation

The dataset of Nefs (2022) is CC BY-SA 4.0; derived data in this repository follow the same licence.
Code licence: _TODO: choose (for example MIT)_.

> Nefs, M. (2022). *Dutch Distribution Centres 2021 Geodata*. 4TU.ResearchData. Dataset.
> https://doi.org/10.4121/19361018
