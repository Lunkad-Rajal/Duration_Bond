# Gap, Shock and Reversal in Indian Corporate Bonds: Pandemic and Duration Checks

This is for an undergraduate course project on Money Banking and Financial Markets. It extends the event design of Kumar (2026), *Delayed Price Discovery and Return Adjustment in Indian Corporate Bonds* (SSRN working paper), using NSE corporate bond trades from January 2019 to August 2026.

Corporate bonds often go days or weeks without trading, so the last price can be out of date. Kumar (2026) reports that longer silences are followed by larger yield jumps, and that large positive jumps partly reverse over the next 20 days. I check whether these two relationships changed with COVID-19, and whether the reversal depends on a bond's duration.

This is an extension of an existing paper on a related but different dataset, not an independent discovery. All results are associations, not causal effects.

---

## Main findings

1. **The 2020 spike in the gap–yield link (H1) is largely a government-yield effect.** The slope of the size of the yield move on log(1 + gap) peaks in the crisis for the raw yield change (0.197 against 0.115 pre-COVID) and for the government yield move itself. For the government-adjusted shock the crisis slope (0.067) is *below* the recovery slope (0.073).
2. **The reversal is present, but only partial, and not specific to silent bonds.** After a shock of 1 percentage point, stale bonds that trade again earn about 0.95 percentage points of duration-adjusted excess return over 20 days (t = 11.9, n = 7,535). In yield terms about 28% of the shock reverses. The same reversal appears for actively traded bonds (gap of 5 days or less), so the slope alone does not show that stale information is the cause.
3. **The crisis period cannot be interpreted for H2.** The pre-COVID and crisis samples are small (379 and 409 events), their confidence intervals include zero, and a shuffled-shock placebo cannot separate the crisis slope from noise (p = 0.114).
4. **The rise of the reversal with duration is almost entirely arithmetic.** The excess return is the yield change multiplied by duration, so a larger price-space slope for long-duration bonds is expected. The shock-by-duration interaction is 0.173 (t = 4.55) and survives security-type fixed effects. But in yield space the reversal is not larger for long-duration bonds (−0.31, −0.26, −0.22 across duration terciles), and the yield-space slope times mean duration reproduces the price-space slopes to within about 6%.

### H1 and H2 by period

<p align="center">
  <img src="figures/fig_h1_h2_by_window.png" alt="H1 and H2 slope estimates by period with 95% confidence intervals" width="95%">
</p>

*Left: H1 slopes by period for four outcomes (raw yield change on all events, raw yield change on events with a shock, government-adjusted shock, government yield move). Right: H2 slopes by period for stale events. Bars are 95% confidence intervals clustered by ISIN. The wide bands in the pre-COVID and crisis periods are the main limit on what can be said about H2 around COVID.*

| H1 slope by period | Pre-COVID | Crisis | Recovery | Post-recovery |
|---|---|---|---|---|
| A. Raw \|ΔY\|, all events | 0.115 | **0.197** | 0.123 | 0.049 |
| C. Government-adjusted \|Shock\| | 0.051 | 0.067 | **0.073** | 0.063 |
| D. Government yield move \|ΔG\| | 0.076 | **0.104** | 0.102 | 0.092 |

A Wald test rejects equal H1 slopes across the four periods for the raw yield change (p < 0.0001) and for the adjusted shock (p = 0.0003). The same test for H2 does not reject (p = 0.37), but with two small samples, failing to reject is weak evidence.

### H2 by duration group

<p align="center">
  <img src="figures/fig_slope_by_duration_group.png" alt="H2 slope by duration group for all, negative and positive shocks" width="85%">
</p>

*H2 slope of the 20-day excess return on the shock by duration group (G1 low, G2 mid, G3 high), for all shocks and separately for negative and positive shocks. Bars are 95% confidence intervals clustered by ISIN. Positive shocks reverse more than negative ones in the low and mid groups; in the high group the difference is not significant (p = 0.23).*

| Duration group | Mean duration (years, approx.) | Price space: ER₂₀ slope | Yield space: residual-yield slope |
|---|---|---|---|
| G1 low | 2.1 | 0.64 | −0.31 |
| G2 mid | 4.5 | 1.10 | −0.26 |
| G3 high | 6.8 | 1.48 | −0.22 |

### Comparison with the original paper

| Regression | Kumar (2026) | This project |
|---|---|---|
| H1: raw \|ΔY\|, all events | 0.079 (n = 308,215) | 0.0799 (n = 311,667) |
| H1: adjusted \|Shock\| | 0.071 (n = 122,933) | 0.0640 (n = 75,920) |
| H2: ER₂₀ on shock, stale events | 0.574 (n = 13,294) | 0.9474 (n = 7,535) |

The raw H1 slope reproduces closely. The H2 slope is larger here. I did not test why; possible reasons are my weekly RBI benchmark, the sample restriction from the security master, and longer durations. The paper's late-sample fall in the adjusted H1 slope is not reproduced.

---

## Method in brief

- **Event:** a trade of a bond (ISIN) paired with its previous trade. **Gap** = calendar days between them. **Stale** = gap ≥ 20 days.
- **Government-adjusted shock** = (bond yield change since the previous trade) − (change in the matched government yield over the same dates). A positive shock means the bond became cheaper relative to government securities.
- **20-day excess return** = −(modified duration) × (residual yield change to the bond's last trade within 20 days). It exists only if the bond trades again within 20 days.
- **Government yield:** midpoint of the weekly minimum and maximum yield on central government securities (RBI/CCIL data) for the fiscal year in which the bond matures, matched to the first weekly observation on or after the trade date within 10 days.
- **Estimation:** OLS with standard errors clustered by ISIN. Breaks across four periods (pre-COVID, crisis, recovery, post-recovery) are tested with Wald tests on period-by-slope interactions. A shuffled-shock placebo (500 shuffles) is used for H2.

The full definitions, equations and verification criteria are in the report.

---

## Data

| Source | Used for | How obtained |
|---|---|---|
| NSE Corporate Bond Market (CBM) daily bhavcopy | Trade prices, yields, traded value (1,609 daily files) | `nse_bond_scraper_v2.py` |
| NSE CBM security master | Maturity date, coupon frequency, issue name (coupon), security type, issuer | NSE reports API (see below) |
| RBI DBIE: Secondary Market Outright Transactions in Government Securities (Face Value) | Government yield by maturity-year bucket (CCIL trade book) | Manual download from the RBI DBIE portal |

Raw data are not included in this repository. All three sources are public.

---

## Reproducing the results

**Requirements:** Python 3.10+ and

```bash
pip install pandas numpy openpyxl curl_cffi
```

The analysis scripts may need further packages; check their imports.

**Steps**

1. **Set paths.** The scripts use hardcoded Windows paths at the top of each file (`RAW_DIR`, `OUT_DIR`, and the start and end dates in the scraper). Edit them for your machine.
2. **Download CBM bhavcopy:** `python nse_bond_scraper_v2.py`. This takes several hours and is rate-limited. NSE blocks plain `requests` calls, so the scraper uses `curl_cffi` with a Chrome fingerprint.
3. **Security master:** if the scraper's automatic download fails, download it in a browser from the NSE reports API and save it in `raw_data` with `master` in the file name. The build script reads any `.csv` or `.xlsx` with `master` in its name.
4. **RBI workbook:** download the series from RBI DBIE and save it in `raw_data` as `Secondary Market Outright Transactions in Government Securities (Face Value).xlsx`.
5. **Build the dataset:** `python build_final_dataset_v3.py`. This writes `events_full.csv` (all events) and `h2_sample.csv` (stale events with a shock and a 20-day return) to the `processed` folder. Check the log for lines starting with `ERROR`; they show which input was missing or not recognised.
6. **Run the analyses** (COVID and duration scripts) to produce the tables and figures.

Please scrape politely and check NSE's terms of use before running the scraper at scale.

---

## Repository structure

```
.
├── nse_bond_scraper_v2.py        # downloads CBM bhavcopy (and the unused WDM files)
├── build_final_dataset_v3.py     # builds events, shocks, duration, forward returns
├── <covid analysis script>.py    # H1/H2 by period, break tests, placebo
├── <duration analysis script>.py # duration groups, interaction model, yield space
├── results_idea1/                # COVID tables and figure
├── results_idea4/                # duration tables and figure
├── figures/
│   ├── fig_h1_h2_by_window.png
│   └── fig_slope_by_duration_group.png
├── report/                       # LaTeX source of the report
└── README.md
```

---

## Limitations

- **Survivorship and calendar time.** The security master is a single September 2026 snapshot that appears to list only bonds still outstanding then. Early-period events can therefore only come from long-maturity bonds, so period comparisons and duration groups are mixed with calendar time to an extent I have not measured.
- **Coverage.** The shock exists for only 24% of events (75,920 of 311,667), and the H2 sample has 7,535 events.
- **Coarse benchmark.** The government yield is weekly, uses the midpoint of a minimum and maximum, and is matched by maturity fiscal year, not exact maturity.
- **Reversal versus noise.** The data cannot separate delayed price discovery from price noise or temporary price pressure at the re-entry trade. Traded-value cuts were not run.
- **Duration is approximate.** Coupons are read from the issue name with fallbacks when missing; convexity and accrued interest are ignored.
- **Inference.** Standard errors are clustered by ISIN, not by date. The pre-COVID and crisis H2 samples are small. Many specifications were examined without correction for multiple testing, and the H2 regressions do not control for the trading gap or traded value.
- **Not tested:** whether yield-space slopes differ significantly across duration groups, and whether active-bond and stale-bond slopes differ.

---

## Reference

Kumar, G. (2026). *Delayed Price Discovery and Return Adjustment in Indian Corporate Bonds.* SSRN working paper.

This project is for research and education only and is not investment advice.
