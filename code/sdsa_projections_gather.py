import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from fredapi import Fred
import os
from pathlib import Path
import statsmodels.api as sm
from scipy import stats                #  for the critical t-value

home = Path.home()
work_dir = (home / 'debt_sustainability_project')
data_dir = (work_dir / 'data' / 'sdsa')
raw_data = (data_dir / 'raw')
clean_data = (data_dir / 'clean')
output = (work_dir / 'output' / 'sdsa' / 'graphics')
code = Path.cwd()

################################################################################
# 1. Define relevant functions
################################################################################
fred = Fred(api_key='8905b2f5faefd705486e644f09bb8088')
def get_fred_series(series_id, series_name):
    """
    Helper function to fetch a series from FRED and return it as a DataFrame.
    """
    data = fred.get_series(series_id)
    df = pd.DataFrame(data, columns=[series_name])
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    return df

def add_recession_bars(ax, recession_df, shortened=False):
    if shortened:
        copy = recession_df[recession_df['date'] >= '1992-01-01']
    else:
        copy = recession_df.copy()
    in_recession = False
    for i in range(len(copy)):
        if copy['recession'].iloc[i] == 1 and not in_recession:
            start_date = copy['date'].iloc[i]
            in_recession = True
        elif copy['recession'].iloc[i] == 0 and in_recession:
            end_date = copy['date'].iloc[i]
            ax.axvspan(start_date, end_date, color='gray', alpha=0.3)
            in_recession = False
    if in_recession:
        end_date = copy['date'].iloc[-1]
        ax.axvspan(start_date, end_date, color='gray', alpha=0.3)

recession_df = get_fred_series('USRECD', 'recession')
filtered_recession_df = recession_df[recession_df['date'] >= '1962-01-01']

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
## Load Budget Lab's Deficit and Debt Projections
################################################################################
def load_tbl_senate_sheet(sheet, header_row, rename_map, flip_sign=False):
    """Load a Budget Lab sheet, reshape wide, rename columns, optionally flip sign, add date."""
    raw_tbl = pd.read_excel(
        raw_data / 'tbl_senate_passed_projections.xlsx',
        sheet_name=sheet, header=header_row
    )
    raw_tbl.rename(columns={'Unnamed: 1': 'variable'}, inplace=True)
    long = raw_tbl.melt(id_vars=['variable'], var_name='date', value_name='value')
    long.dropna(subset=['value'], inplace=True)
    wide = long.pivot_table(index='date', columns='variable', values='value').reset_index()
    wide.rename(columns=rename_map, inplace=True)
    if flip_sign:
        for col in rename_map.values():
            wide[col] *= -1
    wide['date'] = pd.to_datetime(wide['date'].astype(str) + '-01-01')
    return wide[['date'] + list(rename_map.values())].copy()

# F2 = debt levels (positive in source, keep positive)
tbl_senate_b = load_tbl_senate_sheet('F2', 3, {
    'Senate, as written': 'b (tbl senate, as written)',
    'Senate, permanent':  'b (tbl senate, permanent)',
}, flip_sign=False)
master = pd.merge(master, tbl_senate_b, on='date', how='outer')

# F3 = deficits (positive in source = deficit; flip to negative = surplus convention)
tbl_senate_s = load_tbl_senate_sheet('F3', 2, {
    'Senate, as written': 's (tbl senate, as written)',
    'Senate, permanent':  's (tbl senate, permanent)',
}, flip_sign=True)
master = pd.merge(master, tbl_senate_s, on='date', how='outer')

################################################################################
## pull in zandi's tariff scenario projectiosn
################################################################################
zandi_projections = pd.read_excel(
    raw_data / 'zandi_moody_tariff_projections.xlsx',
    sheet_name = 'master',
    header = 0
)
# first, make interest rates real by subtracting pce_deflator
zandi_projections['10_yr_yield'] = (
    zandi_projections['10_yr_yield']
    - zandi_projections['pce_deflator'])
# find real gdp percentage change 
zandi_projections['real gdp'] = zandi_projections['real gdp'].pct_change() * 100
# reshape wide so that we have different columns for each scenario
zandi_projections = (
    zandi_projections.pivot_table(
        index   = "period",            # rows
        columns = "scenario",          # the three scenarios → column level
        values  = ["real gdp", "10_yr_yield"]
    )
)
# ❷ flatten the MultiIndex columns ->  nominalgdp_S1, … 10yryield_S3
zandi_projections.columns = [f"{var} (moody's {sc})" for var, sc in zandi_projections.columns]
zandi_projections.reset_index(inplace=True)
# rename variables 
for var, new_var in zip(['10_yr_yield', 'real gdp'], ['r', 'g']):
    zandi_projections.rename(
        columns = {f"{var} (moody's S1)": f"{new_var} (moody's S1)",
                   f"{var} (moody's S2)": f"{new_var} (moody's S2)",
                   f"{var} (moody's S3)": f"{new_var} (moody's S3)"},
        inplace = True
    )
# convert period into datetime, january 1st
zandi_projections["date"] = (
    pd.PeriodIndex(zandi_projections["period"].astype(str), freq="Q")  # make a PeriodIndex
      .to_timestamp(how="S")                           # S = Start of quarter
)
# drop period
zandi_projections.drop(columns=["period"], inplace=True)
# now merge with master
master = pd.merge(master, zandi_projections, on='date', how='outer') 

################################################################################
## export master dataframe of projections
################################################################################
master.to_csv(clean_data / 'master_projections_cleaned.csv', index=False)