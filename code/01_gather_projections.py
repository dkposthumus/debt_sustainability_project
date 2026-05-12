import pandas as pd
from pathlib import Path

work_dir = Path(__file__).resolve().parent.parent
raw_data = work_dir / 'data' / 'raw'
clean_data = work_dir / 'data' / 'clean'
clean_data.mkdir(parents=True, exist_ok=True)

################################################################################
## Load CBO econ projections
################################################################################
XL_FILE   = f"{raw_data}/cbo_econ_projections.xlsx"
SHEET     = "1. Quarterly"
raw = pd.read_excel(
    XL_FILE,
    sheet_name=SHEET,
    header=None,          # keep *all* rows – we'll discover the header line
    engine="openpyxl"     # avoid the warning on default engine change
)
def _is_quarter_label(x: str) -> bool:
    """return True if x looks like 'YYYYQ#'."""
    return isinstance(x, str) and len(x) == 6 and x.endswith(("Q1","Q2","Q3","Q4"))
hdr_idx = (
    raw
    .apply(lambda row: row.map(_is_quarter_label).any(), axis=1)
    .idxmax()
)                                           # first True -> header row index
header = raw.loc[hdr_idx].tolist()
header[:2] = ["variable", "units"]
data = raw.loc[hdr_idx + 1 :].reset_index(drop=True)
data.columns = header
data["variable"].ffill(inplace=True)
data = data.dropna(subset=["units"]).copy()
quarter_cols = [c for c in data.columns if _is_quarter_label(str(c))]
long = data.melt(
    id_vars = ["variable", "units"],
    value_vars = quarter_cols,
    var_name = "quarter",
    value_name = "value"
)
long["quarter"] = pd.PeriodIndex(long["quarter"], freq="Q")
long["date"] = long["quarter"].dt.to_timestamp(how="S")
tidy = (
    long
    .dropna(subset=["value"])             # get rid of empty cells
    .sort_values(["variable", "date"]) # nice ordering
    .reset_index(drop=True)
)
# restrict to variables we want 
vars_of_interest = [
    '10-Year Treasury note',
    'Chained CPI-U',
    'Gross domestic product (GDP)',
    'GDP price index'
]
tidy = tidy[tidy['variable'].isin(vars_of_interest)].copy()
tidy["series"] = (
    tidy["variable"]
    + " – "
    + tidy["units"].str.lower().str.replace(r"[ ()]", "", regex=True)  # optional cleanup
)
wide = (
    tidy
    .pivot_table(
        index   = "date",
        columns = "series",       # ← single column, so no MultiIndex
        values  = "value",
        aggfunc = "first"
    )
    .reset_index()
)
# restrict only to needed columns
# now convert to real 
wide['r (cbo baseline)'] = (
    wide['10-Year Treasury note – percent']
    - wide['GDP price index – percentagechange,annualrate']
)
wide['g (cbo baseline)'] = (
    wide['Gross domestic product (GDP) – percentagechange,annualrate']
    - wide['GDP price index – percentagechange,annualrate']
)
# now restrict once again 
wide.rename(columns = {'Gross domestic product (GDP) – billionsofdollars': 'gdp (cbo baseline)'}, 
            inplace=True)
# make real 
wide['gdp (cbo baseline)'] = (
    wide['gdp (cbo baseline)'] 
    / (1 + wide['GDP price index – percentagechange,annualrate'] / 100)
)
cbo_econ = wide[['date', 'r (cbo baseline)', 'g (cbo baseline)', 'gdp (cbo baseline)']].copy()

XL_PATH = f"{raw_data}/cbo_budget_projections.xlsx"

def _read_cbo_budget_row(sheet, target_label, occurrence=0):
    """Extract a single row from a CBO budget sheet by matching its label.

    The Feb 2026 CBO file uses a two-row header: 'Actual' on one row, then
    year numbers (2025, 2026, ...) on the next.  We find the year row and
    use it as column names.

    Parameters
    ----------
    occurrence : int
        Which match to use (0-indexed). Useful when the same label appears
        in both "billions of dollars" and "% of GDP" sections.
    """
    raw_b = pd.read_excel(XL_PATH, sheet_name=sheet, header=None, engine="openpyxl")
    # find the row containing year numbers (e.g. 2025, 2026, ...)
    def _has_years(row):
        nums = row.dropna().apply(lambda v: str(v).replace('.0', ''))
        return nums.str.match(r'^\d{4}$').sum() >= 5
    year_row_idx = raw_b.apply(_has_years, axis=1).idxmax()
    # build column names from that row
    year_headers = raw_b.iloc[year_row_idx].tolist()
    year_headers[0] = "variable"
    # data starts after the year row
    df_b = raw_b.iloc[year_row_idx + 1:].copy()
    df_b.columns = year_headers
    df_b = df_b.dropna(subset=["variable"], how="all")
    # skip section headers like "In billions of dollars", "As a percentage of GDP"
    matches = df_b.loc[df_b["variable"].str.contains(target_label, case=False, na=False)]
    row = matches.iloc[[occurrence]]
    row = row.drop(columns=["variable"]).T.reset_index()
    row.columns = ["year_raw", "value"]
    row["year"] = row["year_raw"].astype(str).str.replace(r'\.0$', '', regex=True).str.extract(r"(\d{4})")
    row = row.dropna(subset=["year"])
    row["year"] = row["year"].astype(int)
    row["value"] = pd.to_numeric(row["value"], errors="coerce")
    return row[["year", "value"]].dropna()

# Primary deficit (% of GDP) from Table 1-2 — second occurrence (first is billions)
s_cbo = _read_cbo_budget_row("Table 1-2", "Primary deficit.*adjusted", occurrence=1)
s_cbo.rename(columns={"value": "s (cbo baseline)"}, inplace=True)

# Debt held by the public (% of GDP) from Table 1-3 — first "As a percentage of GDP"
b_cbo = _read_cbo_budget_row("Table 1-3", "As a percentage of GDP", occurrence=0)
b_cbo.rename(columns={"value": "b (cbo baseline)"}, inplace=True)

cbo_budget = s_cbo.merge(b_cbo, on="year")
cbo_budget['date'] = pd.to_datetime(cbo_budget['year'].astype(str) + '-01-01')
master = pd.merge(cbo_econ, cbo_budget, on='date', how='left')

################################################################################
## export master dataframe of projections
################################################################################
master.to_csv(clean_data / 'master_projections_cleaned.csv', index=False)

################################################################################
## Extract AI productivity scenarios from CBO user-specified-projections data
################################################################################
# CBO's Feb 2026 interactive tool (publication 61914, "How Changes in Economic
# Conditions Might Affect the Federal Budget, 2026 to 2036") accepts a
# productivity-growth boost path and reports the implied real GDP growth and
# budget effects via CBO's rules-of-thumb. The tool is web-only (not downloadable);
# we record one transcribed CSV per scenario in data/raw/.
#
# Convention: data/raw/cbo_user_specified_productivity_<label>.csv
#   Required columns:
#     year, real_gdp_growth_user_pct, delta_revenues_bn, delta_mandatory_bn,
#     delta_disc_bn, nominal_gdp_user_bn
#
# The Feb 2026 tool caps the productivity-boost input at +0.5pp/yr, so the 1.0pp
# AI Boom scenario (10pp) is *derived* from the 0.5pp scenario (05pp) via
# linearity: budget effects are linear in (user_nGDP - baseline_nGDP), and the
# real-GDP convolution is approximately linear in the productivity-boost path
# for small boosts. Verified against the Jan 2025 unlocked workbook: the
# year-by-year ratio of 10pp:05pp budget effects is in [2.004, 2.047] — within
# 2.5% of the linear prediction of 2.0× across the 10-yr horizon. We therefore
# scale the (user - baseline) deltas of the 0.5pp scenario by 2× and add them
# back to the Feb 2026 baseline.

def _scenario_from_components(years, g_pct, d_rev, d_mand, d_disc, ngdp_user, label):
    # Change in primary surplus ($bn) = Δrevenues − Δnon-interest outlays.
    # As decimal of GDP (positive = improvement to surplus).
    delta_s_primary = (d_rev - d_mand - d_disc) / ngdp_user
    return pd.DataFrame({
        'year': years,
        'label': label,
        'a_ug_ai': g_pct / 100.0,           # decimal real GDP growth
        'delta_s_primary': delta_s_primary, # decimal of GDP, surplus-positive
    })

def extract_ai_scenario_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment='#')
    return _scenario_from_components(
        years=df['year'].astype(int).values,
        g_pct=df['real_gdp_growth_user_pct'].values,
        d_rev=df['delta_revenues_bn'].values,
        d_mand=df['delta_mandatory_bn'].values,
        d_disc=df['delta_disc_bn'].values,
        ngdp_user=df['nominal_gdp_user_bn'].values,
        label=label,
    )

def _read_cbo_econ_calendar_baseline() -> pd.DataFrame:
    """Read Feb 2026 calendar-year baseline real GDP growth (%) and nominal GDP ($bn)
    from CBO's economic-projections workbook, sheet '2. Calendar Year'."""
    raw_e = pd.read_excel(XL_FILE, sheet_name='2. Calendar Year', header=None,
                           engine='openpyxl')
    # find the row containing year numbers (e.g. 2023, 2024, ...)
    def _has_years(row):
        nums = row.dropna().apply(lambda v: str(v).replace('.0', ''))
        return nums.str.match(r'^\d{4}$').sum() >= 5
    year_row_idx = raw_e.apply(_has_years, axis=1).idxmax()
    headers = raw_e.iloc[year_row_idx].tolist()
    headers[0] = 'variable'
    body = raw_e.iloc[year_row_idx + 1:].copy()
    body.columns = headers
    body['variable'] = body['variable'].ffill()
    # Nominal GDP (Billions of dollars) and Real GDP (Percentage change, annual rate)
    nom = body[(body['variable'].str.contains('Gross domestic product', na=False))
               & (body[headers[1]].astype(str).str.contains('Billions', na=False))].iloc[0]
    rg  = body[(body['variable'] == 'Real GDP')
               & (body[headers[1]].astype(str).str.contains('Percentage change', na=False))].iloc[0]
    years = [c for c in headers if isinstance(c, (int, float)) or
                                    (isinstance(c, str) and c.isdigit())]
    out = pd.DataFrame({
        'year': [int(y) for y in years],
        'baseline_ngdp_cal_bn':  [float(nom[y]) for y in years],
        'baseline_g_pct':        [float(rg[y])  for y in years],
    })
    return out

def derive_10pp_from_05pp(df_05: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Scale (user - baseline) deltas of the 0.5pp scenario by 2× to construct the
    1.0pp scenario, using the Feb 2026 baseline. Approximation: see comment block
    above (within ~2.5% of the workbook formulas over a 10-yr horizon)."""
    df = df_05.merge(baseline, on='year', how='left')
    if df['baseline_ngdp_cal_bn'].isna().any() or df['baseline_g_pct'].isna().any():
        missing = df.loc[df['baseline_ngdp_cal_bn'].isna() | df['baseline_g_pct'].isna(), 'year'].tolist()
        raise ValueError(f"Missing Feb 2026 baseline rows for years: {missing}")
    g_pct_10  = df['baseline_g_pct'] + 2.0 * (df['real_gdp_growth_user_pct'] - df['baseline_g_pct'])
    ngdp_10   = df['baseline_ngdp_cal_bn'] + 2.0 * (df['nominal_gdp_user_bn'] - df['baseline_ngdp_cal_bn'])
    drev_10   = 2.0 * df['delta_revenues_bn']
    dmand_10  = 2.0 * df['delta_mandatory_bn']
    ddisc_10  = 2.0 * df['delta_disc_bn']
    return _scenario_from_components(
        years=df['year'].values, g_pct=g_pct_10.values,
        d_rev=drev_10.values, d_mand=dmand_10.values, d_disc=ddisc_10.values,
        ngdp_user=ngdp_10.values, label='10pp',
    )

# Discover all transcribed CSV scenarios in raw/.
scenario_files = {path.stem.replace('cbo_user_specified_productivity_', ''): path
                  for path in sorted(raw_data.glob('cbo_user_specified_productivity_*.csv'))}
if '05pp' not in scenario_files:
    raise FileNotFoundError(
        f"Required cbo_user_specified_productivity_05pp.csv not found in {raw_data}. "
        f"Transcribe from CBO publication 61914 with inputs1=0,.1,.2,.3,.4,.5,.5,.5,.5,.5,.5."
    )

ai_frames = [extract_ai_scenario_csv(p, label) for label, p in scenario_files.items()]

# Auto-derive the 1.0pp scenario from the 0.5pp web-tool output (Feb 2026 baseline).
if '10pp' not in scenario_files:
    df_05_raw = pd.read_csv(scenario_files['05pp'], comment='#')
    baseline_econ = _read_cbo_econ_calendar_baseline()
    ai_frames.append(derive_10pp_from_05pp(df_05_raw, baseline_econ))
    print("Synthesized '10pp' AI scenario from '05pp' via 2× linearity scaling "
          "against Feb 2026 calendar-year baseline.")

ai_scenarios = pd.concat(ai_frames, ignore_index=True).sort_values(['label', 'year'])
ai_scenarios.to_csv(clean_data / 'ai_scenarios.csv', index=False)
print(f"Wrote {len(ai_scenarios)} rows across "
      f"{ai_scenarios['label'].nunique()} scenarios to ai_scenarios.csv "
      f"(scenarios: {sorted(ai_scenarios['label'].unique())}).")
