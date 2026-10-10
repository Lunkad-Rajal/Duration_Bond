"""
IDEA 1 - COVID WINDOWS  (H1 and H2 across four sub-periods)
============================================================

Follows the written workflow exactly:

  Days 5-6   `window` column: Pre-COVID (Jan 2019-Feb 2020), Crisis (Mar-Dec 2020),
             Recovery (2021-2022), Post-Recovery (2023-2026);
             raw-dY vs residual-shock decomposition for 2020 (year-by-year slopes)
  Days 7-10  H1 and H2 regressions separately on each window;
             formal structural-break test (window dummies x regressor, joint Wald);
             2020 decomposition: raw |dY| vs government-adjusted |shock|
  Days 11-13 SEs clustered by ISIN in EVERY regression;
             shuffled-shock placebo;
             active-bond placebo (gap <= 5) by window;
             N per cell; drop-top-5-ISINs check

Model definitions (assumed from the workflow text - edit below if the paper differs):
  H1 : |dY| (or |shock|) = a + b * log(1 + gap)          all events
  H2 : ER20 = a + b * Shock                               stale bonds, gap >= 20
  Active-bond placebo : ER20 = a + b * Shock              gap <= 5

Input : events_full.csv
Output: CSV tables + summary.txt + one figure, written to --out

Usage:
    python idea1_covid_windows.py --data events_full.csv
    python idea1_covid_windows.py --data events_full.csv --out results_idea1 --n-perm 1000

Requires: pandas, numpy, statsmodels (matplotlib optional, for the figure)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
REQUIRED_COLS = [
    "isin", "date", "gap", "raw_dy", "gov_shock", "g_t", "g_t0", "er_20",
    "vwap_yield", "prev_yield", "vwap_price", "prev_price",
]
WINDOWS = ["1_pre_covid", "2_crisis", "3_recovery", "4_post_recovery"]
SHORT = {"2_crisis": "crisis", "3_recovery": "recovery", "4_post_recovery": "post"}
H2_MIN_GAP = 20
ACTIVE_MAX_GAP = 5
MIN_OBS = 30
MIN_CLUSTERS = 5

LOG = []


def say(msg=""):
    print(msg)
    LOG.append(str(msg))


def show(title, df):
    say("\n=== " + title + " ===")
    if df is None or len(df) == 0:
        say("(no rows)")
        return
    say(df.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


# ----------------------------------------------------------------------------
# Loading and sample construction
# ----------------------------------------------------------------------------
def load_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cannot find '{path}'. Pass the file with --data /full/path/to/events_full.csv"
        )
    wanted = set(REQUIRED_COLS)
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def assign_window(dates):
    """Workflow windows: Pre-COVID < Mar 2020 <= Crisis < 2021 <= Recovery < 2023 <= Post-Recovery."""
    conds = [
        dates < pd.Timestamp("2020-03-01"),
        dates < pd.Timestamp("2021-01-01"),
        dates < pd.Timestamp("2023-01-01"),
    ]
    return np.select(conds, WINDOWS[:3], default=WINDOWS[3])


def build_sample(df):
    flow = [("rows loaded", len(df))]
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    flow.append(("valid dates", len(df)))

    keep = (
        df["vwap_yield"].between(0.5, 20, inclusive="neither")
        & df["prev_yield"].between(0.5, 20, inclusive="neither")
        & df["vwap_price"].between(40, 180, inclusive="neither")
        & df["prev_price"].between(40, 180, inclusive="neither")
        & (df["raw_dy"].abs() < 5)
    )
    df = df[keep].copy()
    flow.append(("after paper filters (yield 0.5-20, price 40-180, |dY|<5pp)", len(df)))

    df = df[df["gap"] >= 1].copy()
    flow.append(("gap >= 1", len(df)))

    df["window"] = assign_window(df["date"])
    df["year"] = df["date"].dt.year
    df["log1p_gap"] = np.log1p(df["gap"].astype(float))
    df["abs_raw"] = df["raw_dy"].abs()
    df["shock"] = df["gov_shock"].astype(float)
    df["abs_shock"] = df["shock"].abs()
    df["dG"] = df["g_t"] - df["g_t0"]
    df["abs_dG"] = df["dG"].abs()
    df["er20"] = df["er_20"].astype(float)
    df["matched"] = df["shock"].notna() & df["dG"].notna()
    return df.reset_index(drop=True), pd.DataFrame(flow, columns=["step", "rows"])


# ----------------------------------------------------------------------------
# Regression helpers (all use SEs clustered by ISIN)
# ----------------------------------------------------------------------------
def fit(d, formula, needed):
    cols = list(dict.fromkeys(list(needed) + ["isin"]))
    sub = d.dropna(subset=cols)
    n = len(sub)
    k = int(sub["isin"].nunique()) if n else 0
    if n < MIN_OBS or k < MIN_CLUSTERS:
        return None, n, k
    groups = pd.factorize(sub["isin"])[0]
    try:
        res = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except (ValueError, np.linalg.LinAlgError):
        return None, n, k
    return res, n, k


def coef_row(res, term, meta, n, k):
    row = dict(meta)
    empty = dict(beta=np.nan, se=np.nan, t=np.nan, p=np.nan,
                 ci_lo=np.nan, ci_hi=np.nan, n=n, clusters=k, r2=np.nan)
    if res is None or term not in res.params.index:
        row.update(empty)
        return row
    ci = res.conf_int().loc[term]
    row.update(
        beta=float(res.params[term]), se=float(res.bse[term]),
        t=float(res.tvalues[term]), p=float(res.pvalues[term]),
        ci_lo=float(ci.iloc[0]), ci_hi=float(ci.iloc[1]),
        n=int(res.nobs), clusters=k, r2=float(res.rsquared),
    )
    return row


def slope(x, y):
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 0:
        return np.nan
    return float(xc @ (y - y.mean()) / denom)


def by_window(d, formula, x, y, meta, extra_needed=()):
    """Run one regression on each window and on all windows pooled."""
    rows = []
    for w in ["ALL"] + WINDOWS:
        sub = d if w == "ALL" else d[d["window"] == w]
        res, n, k = fit(sub, formula, [x, y, *extra_needed])
        rows.append(coef_row(res, x, {**meta, "window": w}, n, k))
    return rows


def break_test(d, x, y, label):
    """Pooled model with window dummies and window x regressor interactions.
    Reports a joint Wald test on (i) the slope interactions and (ii) all dummies + interactions (Chow-type)."""
    sub = d.dropna(subset=[x, y, "isin"]).copy()
    out = {"test": label, "n": len(sub), "clusters": int(sub["isin"].nunique()),
           "slope_chi2": np.nan, "slope_df": np.nan, "slope_p": np.nan,
           "chow_chi2": np.nan, "chow_df": np.nan, "chow_p": np.nan}
    if len(sub) < MIN_OBS or sub["window"].nunique() < 2:
        return out
    dummies, inters = [], []
    for w in WINDOWS[1:]:
        if (sub["window"] == w).sum() < 5:
            continue
        s = SHORT[w]
        sub["d_" + s] = (sub["window"] == w).astype(float)
        sub["x_" + s] = sub[x] * sub["d_" + s]
        dummies.append("d_" + s)
        inters.append("x_" + s)
    if not inters:
        return out
    formula = f"{y} ~ {x} + " + " + ".join(dummies + inters)
    groups = pd.factorize(sub["isin"])[0]
    try:
        res = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": groups})
        w1 = res.wald_test(", ".join(f"{t} = 0" for t in inters), use_f=False, scalar=True)
        w2 = res.wald_test(", ".join(f"{t} = 0" for t in dummies + inters), use_f=False, scalar=True)
    except (ValueError, np.linalg.LinAlgError):
        return out
    out.update(
        slope_chi2=float(np.squeeze(w1.statistic)), slope_df=len(inters), slope_p=float(np.squeeze(w1.pvalue)),
        chow_chi2=float(np.squeeze(w2.statistic)), chow_df=len(dummies + inters), chow_p=float(np.squeeze(w2.pvalue)),
    )
    return out


def shuffle_placebo(d, n_perm, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for w in ["ALL"] + WINDOWS:
        sub = d if w == "ALL" else d[d["window"] == w]
        sub = sub.dropna(subset=["shock", "er20"])
        if len(sub) < MIN_OBS:
            rows.append(dict(window=w, n=len(sub), actual_beta=np.nan, placebo_mean=np.nan,
                             placebo_sd=np.nan, placebo_p95_abs=np.nan, empirical_p=np.nan))
            continue
        x = sub["shock"].to_numpy(dtype=float)
        y = sub["er20"].to_numpy(dtype=float)
        actual = slope(x, y)
        draws = np.array([slope(rng.permutation(x), y) for _ in range(n_perm)])
        draws = draws[np.isfinite(draws)]
        if len(draws) == 0 or not np.isfinite(actual):
            emp_p, pm, ps, p95 = (np.nan,) * 4
        else:
            emp_p = (1 + int(np.sum(np.abs(draws) >= abs(actual)))) / (1 + len(draws))
            pm, ps = float(draws.mean()), float(draws.std())
            p95 = float(np.quantile(np.abs(draws), 0.95))
        rows.append(dict(window=w, n=len(sub), actual_beta=actual, placebo_mean=pm,
                         placebo_sd=ps, placebo_p95_abs=p95, empirical_p=emp_p))
    return rows


def make_figure(h1, h2, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        say("(matplotlib not installed - figure skipped)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    specs = h1["spec"].drop_duplicates().tolist()
    for i, s in enumerate(specs):
        tab = h1[(h1["spec"] == s) & (h1["window"] != "ALL")].dropna(subset=["beta"])
        if tab.empty:
            continue
        xs = np.arange(len(tab)) + (i - (len(specs) - 1) / 2) * 0.18
        err = np.vstack([tab["beta"] - tab["ci_lo"], tab["ci_hi"] - tab["beta"]])
        ax.errorbar(xs, tab["beta"], yerr=err, fmt="o", capsize=3, label=s)
        ax.set_xticks(np.arange(len(tab)))
        ax.set_xticklabels(tab["window"].tolist(), rotation=20)
    ax.set_title("H1: slope of |move| on log(1+gap)")
    ax.legend(fontsize=7)
    ax = axes[1]
    tab = h2[h2["window"] != "ALL"].dropna(subset=["beta"])
    if not tab.empty:
        xs = np.arange(len(tab))
        err = np.vstack([tab["beta"] - tab["ci_lo"], tab["ci_hi"] - tab["beta"]])
        ax.errorbar(xs, tab["beta"], yerr=err, fmt="o", capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(tab["window"].tolist(), rotation=20)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("H2: slope of ER20 on shock (gap >= 20)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Idea 1: COVID windows for H1 and H2")
    ap.add_argument(
        "--data",
        default=r"C:\A - Personal\User\UnderGrad\MBFM Project\Datasets\processed\events_full.csv",
        help="events_full.csv",
    )
    ap.add_argument("--out", default="results_idea1", help="output folder")
    ap.add_argument("--n-perm", type=int, default=500, help="permutations for the shuffled-shock placebo")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    raw = load_data(args.data)
    d, flow = build_sample(raw)
    if len(d) < MIN_OBS:
        raise ValueError(f"Only {len(d)} usable rows after filtering - cannot proceed.")
    show("Sample construction", flow)

    # integrity check: gov_shock should equal raw_dy - (g_t - g_t0)
    m = d[d["matched"]]
    if len(m):
        dev = float((m["shock"] - (m["raw_dy"] - m["dG"])).abs().max())
        say(f"\nIntegrity check: max |gov_shock - (raw_dy - (g_t - g_t0))| = {dev:.2e}")

    h2 = d[(d["gap"] >= H2_MIN_GAP) & d["shock"].notna() & d["er20"].notna()]
    active = d[(d["gap"] <= ACTIVE_MAX_GAP) & d["shock"].notna() & d["er20"].notna()]
    matched = d[d["matched"]]

    # --- N per cell ------------------------------------------------------------
    cells = []
    for w in WINDOWS:
        sub = d[d["window"] == w]
        cells.append({
            "window": w,
            "all_events": len(sub), "all_isins": int(sub["isin"].nunique()),
            "with_shock": int(sub["matched"].sum()),
            "H2_n": int(((sub["gap"] >= H2_MIN_GAP) & sub["shock"].notna() & sub["er20"].notna()).sum()),
            "H2_isins": int(sub.loc[(sub["gap"] >= H2_MIN_GAP) & sub["shock"].notna() & sub["er20"].notna(), "isin"].nunique()),
            "active_placebo_n": int(((sub["gap"] <= ACTIVE_MAX_GAP) & sub["shock"].notna() & sub["er20"].notna()).sum()),
        })
    t_cells = pd.DataFrame(cells)
    show("N per cell (window)", t_cells)

    # --- Days 7-10: H1 by window ------------------------------------------------
    rows = []
    rows += by_window(d, "abs_raw ~ log1p_gap", "log1p_gap", "abs_raw",
                      {"spec": "A: raw |dY| (all events)"})
    rows += by_window(matched, "abs_raw ~ log1p_gap", "log1p_gap", "abs_raw",
                      {"spec": "B: raw |dY| (events with shock)"})
    rows += by_window(matched, "abs_shock ~ log1p_gap", "log1p_gap", "abs_shock",
                      {"spec": "C: adjusted |shock|"})
    rows += by_window(matched, "abs_dG ~ log1p_gap", "log1p_gap", "abs_dG",
                      {"spec": "D: G-sec |dG| (events with shock)"})
    t_h1 = pd.DataFrame(rows)
    show("H1: |move| = a + b*log(1+gap), by window (ISIN-clustered)", t_h1)

    # --- Days 7-10: H2 by window ------------------------------------------------
    t_h2 = pd.DataFrame(by_window(h2, "er20 ~ shock", "shock", "er20",
                                  {"spec": "ER20 ~ Shock (gap >= 20)"}))
    show("H2: ER20 = a + b*Shock, by window (ISIN-clustered)", t_h2)

    # --- Days 7-10: structural-break tests --------------------------------------
    t_break = pd.DataFrame([
        break_test(d, "log1p_gap", "abs_raw", "H1 raw |dY| (all events)"),
        break_test(matched, "log1p_gap", "abs_shock", "H1 adjusted |shock|"),
        break_test(h2, "shock", "er20", "H2 ER20 on shock"),
    ])
    show("Structural break: Wald tests on window interactions (base = pre-COVID)", t_break)

    # --- Days 5-10: 2020 decomposition (year-by-year) ---------------------------
    yrows = []
    for yr in sorted(d["year"].unique()):
        ys = d[d["year"] == yr]
        ym = matched[matched["year"] == yr]
        row = {"year": int(yr)}
        for label, data, formula, x, y in [
            ("slope_rawY_all", ys, "abs_raw ~ log1p_gap", "log1p_gap", "abs_raw"),
            ("slope_rawY_matched", ym, "abs_raw ~ log1p_gap", "log1p_gap", "abs_raw"),
            ("slope_adj_shock", ym, "abs_shock ~ log1p_gap", "log1p_gap", "abs_shock"),
            ("slope_dG", ym, "abs_dG ~ log1p_gap", "log1p_gap", "abs_dG"),
        ]:
            res, n, k = fit(data, formula, [x, y])
            row[label] = float(res.params[x]) if res is not None else np.nan
            if label == "slope_rawY_matched":
                row["n_matched"] = n
        raw_m, adj = row["slope_rawY_matched"], row["slope_adj_shock"]
        row["adj_over_raw_matched"] = adj / raw_m if (np.isfinite(raw_m) and raw_m != 0) else np.nan
        yrows.append(row)
    t_year = pd.DataFrame(yrows)
    show("H1 slopes by year: raw vs government-adjusted (2020 decomposition)", t_year)

    # 2020-only comparison on the SAME events, with SEs
    d20 = matched[matched["year"] == 2020]
    dec = []
    for label, formula, y in [("raw |dY|", "abs_raw ~ log1p_gap", "abs_raw"),
                              ("G-sec |dG|", "abs_dG ~ log1p_gap", "abs_dG"),
                              ("adjusted |shock|", "abs_shock ~ log1p_gap", "abs_shock")]:
        res, n, k = fit(d20, formula, ["log1p_gap", y])
        dec.append(coef_row(res, "log1p_gap", {"year": 2020, "dependent": label}, n, k))
    t_dec = pd.DataFrame(dec)
    show("2020 only, same events: raw vs G-sec vs residual", t_dec)

    # --- Days 11-13: robustness -------------------------------------------------
    t_plc = pd.DataFrame(shuffle_placebo(h2, args.n_perm, args.seed))
    show(f"Shuffled-shock placebo on H2 ({args.n_perm} permutations)", t_plc)

    t_act = pd.DataFrame(by_window(active, "er20 ~ shock", "shock", "er20",
                                   {"spec": f"Active-bond placebo (gap <= {ACTIVE_MAX_GAP})"}))
    show("Active-bond placebo by window (should stay flat)", t_act)

    top5 = h2["isin"].value_counts().head(5).index.tolist()
    say(f"\nLargest 5 ISINs in the H2 sample (dropped below): {top5}")
    t_drop = pd.DataFrame(by_window(h2[~h2["isin"].isin(top5)], "er20 ~ shock", "shock", "er20",
                                    {"spec": "H2, top-5 ISINs dropped"}))
    show("Robustness: H2 with the 5 largest ISINs dropped", t_drop)

    # --- Save --------------------------------------------------------------------
    tables = {
        "t0_sample_flow.csv": flow, "t1_cell_counts.csv": t_cells,
        "t2_h1_by_window.csv": t_h1, "t3_h2_by_window.csv": t_h2,
        "t4_break_tests.csv": t_break, "t5_h1_by_year.csv": t_year,
        "t5_decomposition_2020.csv": t_dec, "t6_shuffle_placebo.csv": t_plc,
        "t7_active_placebo.csv": t_act, "t8_h2_drop_top5_isins.csv": t_drop,
    }
    for name, tab in tables.items():
        tab.to_csv(os.path.join(args.out, name), index=False)
    make_figure(t_h1, t_h2, os.path.join(args.out, "fig_h1_h2_by_window.png"))
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    say(f"\nDone. Tables, figure and summary.txt written to: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)