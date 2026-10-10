"""
IDEA 4 - DURATION HETEROGENEITY  (H2 sample: stale bonds, gap >= 20 days)
==========================================================================

Follows the written workflow exactly:

  Days 5-6   form duration terciles (or quintiles) on the H2 sample and build
             the interaction term  Shock x Duration
  Days 7-10  (a) ER20 = a + b*Shock, run separately per duration group
             (b) ER20 = a + b1*Shock + b2*(Shock x Duration) + b3*Duration
             (c) asymmetry: positive vs negative shocks within each group
  Days 11-13 (a) SEs clustered by ISIN in EVERY regression
             (b) shuffled-shock placebo
             (c) N per cell
             (d) terciles vs security type (re-run within sec_type)
             (e) concentration check: drop the 5 largest issuers
             (f) investment-grade-only re-run -> NOT possible: the files have
                 no rating column (the workflow says "if you have coverage")

Input : h2_sample.csv  (or events_full.csv - the H2 filter is re-applied, so
        either file gives the same sample)
Output: CSV tables + summary.txt + one figure, written to --out

Usage:
    python idea4_duration_heterogeneity.py --data h2_sample.csv
    python idea4_duration_heterogeneity.py --data h2_sample.csv --n-groups 5 --out results_q5

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
    "isin", "issuer", "sec_type", "gap", "raw_dy", "gov_shock", "er_20",
    "duration", "vwap_yield", "prev_yield", "vwap_price", "prev_price",
]
OPTIONAL_COLS = ["fwd_resid_dy_20"]      # only used for the yield-space table
MIN_GAP = 20                              # H2 definition (stale bonds)
MIN_OBS = 30                              # smallest sample a regression is run on
MIN_CLUSTERS = 5                          # smallest number of ISIN clusters
MIN_SEC_TYPE_OBS = 50                     # rarer sec_types are pooled into 'OTHER'

LOG = []


def say(msg=""):
    """Print and remember a line for summary.txt."""
    print(msg)
    LOG.append(str(msg))


def show(title, df):
    """Pretty-print a table."""
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
            f"Cannot find '{path}'. Pass the file with --data /full/path/to/h2_sample.csv"
        )
    wanted = set(REQUIRED_COLS + OPTIONAL_COLS)
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def build_h2_sample(df):
    """Paper filters + H2 definition. Returns (sample, flow_table)."""
    flow = [("rows loaded", len(df))]

    keep = (
        df["vwap_yield"].between(0.5, 20, inclusive="neither")
        & df["prev_yield"].between(0.5, 20, inclusive="neither")
        & df["vwap_price"].between(40, 180, inclusive="neither")
        & df["prev_price"].between(40, 180, inclusive="neither")
        & (df["raw_dy"].abs() < 5)
    )
    df = df[keep].copy()
    flow.append(("after paper filters (yield 0.5-20, price 40-180, |dY|<5pp)", len(df)))

    df = df[df["gap"] >= MIN_GAP]
    flow.append((f"gap >= {MIN_GAP}", len(df)))

    df = df[df["gov_shock"].notna()]
    flow.append(("non-missing gov_shock", len(df)))

    df = df[df["er_20"].notna()]
    flow.append(("non-missing er_20", len(df)))

    df = df[df["duration"].notna() & (df["duration"] > 0)]
    flow.append(("non-missing, positive duration", len(df)))

    df = df.copy()
    df["shock"] = df["gov_shock"].astype(float)
    df["er20"] = df["er_20"].astype(float)
    df["pos"] = (df["shock"] > 0).astype(int)
    df["side"] = np.where(df["pos"] == 1, "positive", "negative")
    df["shock_x_pos"] = df["shock"] * df["pos"]
    df["shock_x_dur"] = df["shock"] * df["duration"]
    df["dur_c"] = df["duration"] - df["duration"].mean()
    df["shock_x_durc"] = df["shock"] * df["dur_c"]
    df["issuer"] = df["issuer"].fillna("UNKNOWN").astype(str)
    df["sec_type"] = df["sec_type"].fillna("UNK").astype(str)

    counts = df["sec_type"].value_counts()
    rare = set(counts[counts < MIN_SEC_TYPE_OBS].index)
    df["sec_grp"] = df["sec_type"].where(~df["sec_type"].isin(rare), "OTHER").astype(object)

    flow_df = pd.DataFrame(flow, columns=["step", "rows"])
    return df.reset_index(drop=True), flow_df


def group_names(k):
    if k == 3:
        return ["G1_low", "G2_mid", "G3_high"]
    return [f"G{i + 1}" + ("_low" if i == 0 else "_high" if i == k - 1 else "") for i in range(k)]


def assign_groups(d, k):
    """Quantile groups of duration (terciles for k=3, quintiles for k=5)."""
    codes = pd.qcut(d["duration"], k, labels=False, duplicates="drop")
    n_bins = int(codes.max()) + 1
    names = group_names(n_bins)
    mapping = {i: names[i] for i in range(n_bins)}
    return codes.map(mapping).astype(object), names


# ----------------------------------------------------------------------------
# Regression helpers (all use SEs clustered by ISIN)
# ----------------------------------------------------------------------------
def fit(d, formula, needed):
    """OLS with ISIN-clustered SEs. Returns (result_or_None, n_obs, n_clusters)."""
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
    """One tidy row for a single coefficient."""
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
    """Plain OLS slope (used only for the fast permutation placebo)."""
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 0:
        return np.nan
    return float(xc @ (y - y.mean()) / denom)


def subset(d, gcol, g):
    return d if g == "ALL" else d[d[gcol] == g]


# ----------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------
def cell_counts(d, gcol):
    out = (
        d.groupby([gcol, "side"])
        .agg(n=("isin", "size"), isins=("isin", "nunique"), issuers=("issuer", "nunique"),
             mean_duration=("duration", "mean"), mean_shock=("shock", "mean"))
        .reset_index()
        .rename(columns={gcol: "group"})
    )
    return out


def group_regressions(d, gcol, groups, y="er20", x="shock", meta=None):
    rows = []
    for g in ["ALL"] + list(groups):
        sub = subset(d, gcol, g)
        res, n, k = fit(sub, f"{y} ~ {x}", [y, x])
        m = {"group": g}
        if meta:
            m = {**meta, **m}
        rows.append(coef_row(res, x, m, n, k))
    return rows


def interacted_models(d):
    specs = [
        ("raw duration",
         "er20 ~ shock + shock_x_dur + duration",
         ["er20", "shock", "shock_x_dur", "duration"],
         ["shock", "shock_x_dur", "duration"]),
        ("mean-centred duration",
         "er20 ~ shock + shock_x_durc + dur_c",
         ["er20", "shock", "shock_x_durc", "dur_c"],
         ["shock", "shock_x_durc", "dur_c"]),
        ("mean-centred duration + sec_type FE",
         "er20 ~ shock + shock_x_durc + dur_c + C(sec_grp)",
         ["er20", "shock", "shock_x_durc", "dur_c", "sec_grp"],
         ["shock", "shock_x_durc", "dur_c"]),
    ]
    rows = []
    for label, formula, needed, terms in specs:
        res, n, k = fit(d, formula, needed)
        for t in terms:
            rows.append(coef_row(res, t, {"model": label, "term": t}, n, k))
    return rows


def asymmetry(d, gcol, groups, y="er20", meta=None):
    rows = []
    for g in ["ALL"] + list(groups):
        sub = subset(d, gcol, g)
        base = {"group": g}
        if meta:
            base = {**meta, **base}
        for side, flag in (("negative", 0), ("positive", 1)):
            part = sub[sub["pos"] == flag]
            res, n, k = fit(part, f"{y} ~ shock", [y, "shock"])
            rows.append(coef_row(res, "shock", {**base, "side": side}, n, k))
        # formal test: is the positive-shock slope different from the negative-shock slope?
        res, n, k = fit(sub, f"{y} ~ shock + pos + shock_x_pos", [y, "shock", "pos", "shock_x_pos"])
        rows.append(coef_row(res, "shock_x_pos", {**base, "side": "difference (pos - neg)"}, n, k))
    return rows


def shuffle_placebo(d, gcol, groups, n_perm, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for g in ["ALL"] + list(groups):
        sub = subset(d, gcol, g).dropna(subset=["shock", "er20"])
        if len(sub) < MIN_OBS:
            rows.append(dict(group=g, n=len(sub), actual_beta=np.nan, placebo_mean=np.nan,
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
        rows.append(dict(group=g, n=len(sub), actual_beta=actual, placebo_mean=pm,
                         placebo_sd=ps, placebo_p95_abs=p95, empirical_p=emp_p))
    return rows


def within_sec_type(d, k_groups, min_n):
    """Re-cut duration groups INSIDE each large sec_type, then run the group regressions."""
    rows = []
    sizes = d["sec_grp"].value_counts()
    for st in [s for s in sizes.index if sizes[s] >= min_n and s != "OTHER"]:
        sub = d[d["sec_grp"] == st].copy()
        if len(sub) < k_groups * 50:
            continue
        sub["dur_grp_in"], names = assign_groups(sub, k_groups)
        rows += group_regressions(sub, "dur_grp_in", names, meta={"sec_type": st})
    return rows


def make_figure(t_groups, t_asym, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        say("(matplotlib not installed - figure skipped)")
        return
    series = [
        ("all shocks", t_groups[t_groups["group"] != "ALL"]),
        ("negative shocks", t_asym[(t_asym["side"] == "negative") & (t_asym["group"] != "ALL")]),
        ("positive shocks", t_asym[(t_asym["side"] == "positive") & (t_asym["group"] != "ALL")]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, tab) in enumerate(series):
        tab = tab.dropna(subset=["beta"])
        if tab.empty:
            continue
        xs = np.arange(len(tab)) + (i - 1) * 0.15
        err = np.vstack([tab["beta"] - tab["ci_lo"], tab["ci_hi"] - tab["beta"]])
        ax.errorbar(xs, tab["beta"], yerr=err, fmt="o", capsize=3, label=label)
        ax.set_xticks(np.arange(len(tab)))
        ax.set_xticklabels(tab["group"].tolist())
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("slope of ER20 on shock (95% CI, ISIN-clustered)")
    ax.set_title("H2 drift by duration group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Idea 4: duration heterogeneity in the H2 sample")
    ap.add_argument(
        "--data",
        default=r"C:\A - Personal\User\UnderGrad\MBFM Project\Datasets\processed\h2_sample.csv",
        help="h2_sample.csv or events_full.csv",
    )
    ap.add_argument("--out", default="results_idea4", help="output folder")
    ap.add_argument("--n-groups", type=int, default=3, help="3 = terciles, 5 = quintiles")
    ap.add_argument("--n-perm", type=int, default=500, help="permutations for the shuffled-shock placebo")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)

    if args.n_groups < 2:
        raise ValueError("--n-groups must be at least 2")
    os.makedirs(args.out, exist_ok=True)

    raw = load_data(args.data)
    d, flow = build_h2_sample(raw)
    if len(d) < MIN_OBS:
        raise ValueError(f"Only {len(d)} usable H2 rows after filtering - cannot proceed.")
    d["dur_grp"], groups = assign_groups(d, args.n_groups)

    show("Sample construction", flow)
    say(f"\nH2 sample: {len(d):,} events, {d['isin'].nunique():,} ISINs, "
        f"{d['issuer'].nunique():,} issuers, {args.n_groups} duration groups")
    cuts = d.groupby("dur_grp")["duration"].agg(["min", "max", "count"]).reset_index()
    show("Duration group boundaries (years)", cuts)

    # --- Days 11-13: N per cell --------------------------------------------
    t_cells = cell_counts(d, "dur_grp")
    show("N per cell (duration group x shock sign)", t_cells)

    # --- Days 7-10 (a): per-group regression --------------------------------
    t_groups = pd.DataFrame(group_regressions(d, "dur_grp", groups, meta={"spec": "ER20 ~ Shock"}))
    show("(a) ER20 = a + b*Shock, by duration group (ISIN-clustered)", t_groups)

    # --- Days 7-10 (b): fully interacted ------------------------------------
    t_inter = pd.DataFrame(interacted_models(d))
    show("(b) Interacted model: ER20 = a + b1*Shock + b2*(Shock x Duration) + b3*Duration", t_inter)

    # --- Days 7-10 (c): asymmetry -------------------------------------------
    t_asym = pd.DataFrame(asymmetry(d, "dur_grp", groups))
    show("(c) Asymmetry: positive vs negative shocks within each duration group", t_asym)

    # --- Days 11-13: robustness ---------------------------------------------
    t_plc = pd.DataFrame(shuffle_placebo(d, "dur_grp", groups, args.n_perm, args.seed))
    show(f"Shuffled-shock placebo ({args.n_perm} permutations)", t_plc)

    t_sec = pd.DataFrame(within_sec_type(d, args.n_groups, min_n=300))
    show("Robustness: group regressions re-cut WITHIN each large sec_type", t_sec)

    sec_mix = pd.crosstab(d["sec_grp"], d["dur_grp"]).reset_index()
    show("Security type x duration group (is duration proxying for sec_type?)", sec_mix)

    top5 = d["issuer"].value_counts().head(5).index.tolist()
    d_drop = d[~d["issuer"].isin(top5)]
    say(f"\nDropping the 5 largest issuers: {top5}  ({len(d) - len(d_drop):,} rows removed)")
    t_drop = pd.DataFrame(group_regressions(d_drop, "dur_grp", groups,
                                            meta={"spec": "ER20 ~ Shock, top-5 issuers dropped"}))
    show("Robustness: top-5 issuers dropped", t_drop)

    say("\nNOTE: investment-grade-only re-run skipped - there is no rating column in the files "
        "(the workflow makes this check conditional on coverage).")

    t_yield = pd.DataFrame()
    if "fwd_resid_dy_20" in d.columns:
        t_yield = pd.DataFrame(group_regressions(
            d, "dur_grp", groups, y="fwd_resid_dy_20",
            meta={"spec": "fwd_resid_dy_20 ~ Shock (yield space)"}))
        show("Supplement: same regression in yield space (ER20 = -duration x fwd_resid_dy_20)", t_yield)

    # --- Save -----------------------------------------------------------------
    tables = {
        "t0_sample_flow.csv": flow, "t0_duration_cuts.csv": cuts,
        "t1_cell_counts.csv": t_cells, "t2_group_regressions.csv": t_groups,
        "t3_interacted_models.csv": t_inter, "t4_asymmetry.csv": t_asym,
        "t5_shuffle_placebo.csv": t_plc, "t6_within_sectype.csv": t_sec,
        "t6_sectype_by_group.csv": sec_mix, "t7_drop_top5_issuers.csv": t_drop,
        "t8_yield_space.csv": t_yield,
    }
    for name, tab in tables.items():
        tab.to_csv(os.path.join(args.out, name), index=False)
    make_figure(t_groups, t_asym, os.path.join(args.out, "fig_slope_by_duration_group.png"))
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