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
data = (work_dir / 'data' / 'sdsa')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
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
    header=None,          # keep *all* rows – we’ll discover the header line
    engine="openpyxl"     # avoid the warning on default engine change
)
def _is_quarter_label(x: str) -> bool:
    """return True if x looks like ‘YYYYQ#’."""
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

XL_PATH   = f"{raw_data}/cbo_budget_projections.xlsx"
TAB_NAME  = "Table B-1"        # adjust if the sheet is named differently
TARGETS   = ["Primary deficit (-)", "Debt held by the public"]
raw = pd.read_excel(XL_PATH, sheet_name=TAB_NAME, header=None, engine="openpyxl")
# — find the first row that contains the word “Actual” ————————
hdr_row = (
    raw.apply(lambda r: r.astype(str).str.contains("Actual", case=False).any(), axis=1)
       .idxmax()
)
df = pd.read_excel(
    XL_PATH,
    sheet_name=TAB_NAME,
    skiprows=hdr_row,   # throw away everything above the header
    header=0,           # Excel row hdr_row becomes the column names
    engine="openpyxl"
)
# first column contains the item names
df.rename(columns={df.columns[0]: "variable"}, inplace=True)
two = (
    df.loc[df["variable"].isin(TARGETS)]
      .set_index("variable")            # index = the two variable names
      .T                                # years become the index
      .rename_axis("year")
      .reset_index()
)
# column “year” has items like “Actual, 2024”, “2025”, … → extract YYYY
two["year"] = two["year"].astype(str).str.extract(r"(\d{4})").astype(int)
cbo_budget = (
    two
    .set_index("year")            # optional: make year the index
    .sort_index()
    [TARGETS]                     # ensure column order
)
dup_idx = [i for i, c in enumerate(cbo_budget.columns)
           if c == "Primary deficit (-)"]
# say we want to rename the SECOND copy (index 1 of that list)
cols = cbo_budget.columns.tolist()      # to a mutable Python list
cols[dup_idx[1]] = "s (cbo baseline)"   # new name
cbo_budget.columns = cols  # put the list back
# repeat for debt_public
dup_idx = [i for i, c in enumerate(cbo_budget.columns)
           if c == "Debt held by the public"]
cols[dup_idx[1]] = "b (cbo baseline)"   # new name
# now we have the columns we want, but the first copy is still there
# put the list back
cbo_budget.columns = cols
cbo_budget.reset_index(inplace=True)  # make year a column again
cbo_budget = cbo_budget[['year', 's (cbo baseline)', 'b (cbo baseline)']].copy()
# now merge cbo_econ and cbo_budget
cbo_budget['date'] = pd.to_datetime(cbo_budget['year'].astype(str) + '-01-01')
master = pd.merge(cbo_econ, cbo_budget, on='date', how='left')

################################################################################
## Load Budget Lab's Deficit and Debt Projections
################################################################################
raw = pd.read_excel(
    raw_data / 'tbl_senate_passed_projections.xlsx',
    sheet_name = 'F2',
    header = 3
)
raw.rename(columns = {
    'Unnamed: 1': 'variable'
}, inplace=True)
# reshape long, so that variable are the columns
long = raw.melt(
    id_vars = ['variable'],
    var_name = 'date',
    value_name = 'value'
)
# drop instances where value is nan
long.dropna(subset=['value'], inplace=True)
# reshape wide 
wide = long.pivot_table(
    index = 'date',
    columns = 'variable',
    values = 'value'
).reset_index()
wide.rename(columns = {
    'Senate, as written': 'b (tbl senate, as written)',
    'Senate, permanent': 'b (tbl senate, permanent)',
}, inplace=True)
# each of these are reversed, so multiply by -1
wide['b (tbl senate, as written)'] *= -1
wide['b (tbl senate, permanent)'] *= -1
# convert date into datetime, january 1st
wide['date'] = pd.to_datetime(wide['date'].astype(str) + '-01-01')
tbl_senate_b = wide[['date', 'b (tbl senate, as written)', 'b (tbl senate, permanent)']].copy()
# now merge with cbo_econ
master = pd.merge(master, tbl_senate_b, on='date', how='outer')

# now pull in deficit projections
raw = pd.read_excel(
    raw_data / 'tbl_senate_passed_projections.xlsx',
    sheet_name = 'F3',
    header = 2
)
raw.rename(columns = {
    'Unnamed: 1': 'variable'
}, inplace=True)
# reshape long, so that variable are the columns
long = raw.melt(
    id_vars = ['variable'],
    var_name = 'date',
    value_name = 'value'
)
# drop instances where value is nan
long.dropna(subset=['value'], inplace=True)
# reshape wide 
wide = long.pivot_table(
    index = 'date',
    columns = 'variable',
    values = 'value'
).reset_index()
wide.rename(columns = {
    'Senate, as written': 's (tbl senate, as written)',
    'Senate, permanent': 's (tbl senate, permanent)',
}, inplace=True)
# each of these are reversed, so multiply by -1 
wide['s (tbl senate, as written)'] *= -1
wide['s (tbl senate, permanent)'] *= -1
# convert date into datetime, january 1st
wide['date'] = pd.to_datetime(wide['date'].astype(str) + '-01-01')
tbl_senate_s = wide[['date', 's (tbl senate, as written)', 's (tbl senate, permanent)']].copy()
# now merge with cbo_econ
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