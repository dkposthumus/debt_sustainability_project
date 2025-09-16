import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from fredapi import Fred
import os
from pathlib import Path
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from scipy import stats                #  for the critical t-value

home = Path.home()
work_dir = (home / 'debt_sustainability_project')
data = (work_dir / 'data' / 'sdsa')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
output = (work_dir / 'output' / 'sdsa' / 'graphics')
code = Path.cwd() 

################################################################################
# Define relevant functions
################################################################################
# set matplotlib style 
plt.style.use('mahoney_lab.mplstyle')

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

def add_footnote_text(source_text):
    foot_y = 0
    plt.figtext(0.01, foot_y, "Sources:", fontfamily="Times New Roman",
                fontweight="bold", fontsize=8, ha="left", va="bottom")
    plt.figtext(0.075, foot_y, source_text, fontfamily="Times New Roman",
                fontsize=8, ha="left", va="bottom")

recession_df = get_fred_series('USRECD', 'recession')
filtered_recession_df = recession_df[recession_df['date'] >= '1962-01-01']

################################################################################
## Load Data - SPF vs. BEA estimes of r and g
################################################################################
r_g_master = pd.read_csv(clean_data / 'r_g_master.csv', parse_dates=['date'])
# now r_g_master does NOT have real growth rate, instead just nominal growth rate
# let's bring in gdp_deflator 
gdp_deflator = get_fred_series('GDPDEF', 'GDP Deflator')
# collapse on year 
gdp_deflator['date'] = gdp_deflator['date'].dt.to_period('Y').dt.to_timestamp()
# collapse mean 
gdp_deflator = gdp_deflator.groupby('date').mean().reset_index()
# convert to percentage change
gdp_deflator['GDP Deflator'] = gdp_deflator['GDP Deflator'].pct_change() * 100
# rename column
gdp_deflator.rename(columns={'GDP Deflator': 'gdp_deflator'}, inplace=True)
# now merge gdp_deflator into r_g_master
r_g_master = pd.merge(r_g_master, gdp_deflator, on='date', how='left')
# now plot for comparison
for col in ['r', 'g']:
    plt.figure(figsize=(12, 6))
    plt.plot(r_g_master['date'], r_g_master[f'{col} (SPF / Blanchard)'], 
             label=f'{col} (SPF/Blanchard)', 
             color='orange', linestyle='-')
    # forward fill BEA/Historical data for and r and g and make real
    r_g_master[f'{col} (BEA / Jared)'] = (r_g_master[f'{col} (BEA / Jared)'] 
                                          - r_g_master['gdp_deflator'])
    r_g_master[f'{col} (BEA / Jared)'] = r_g_master[f'{col} (BEA / Jared)'].ffill()
    plt.plot(r_g_master['date'], r_g_master[f'{col} (BEA / Jared)'], 
             color='blue', linestyle='--', label=f'{col} (BEA Historical Data)')
    # add recession bars
    add_recession_bars(plt.gca(), filtered_recession_df, shortened=False)
    # add horizontal line at y = 0
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f'Comparison of SPF and BEA Estimates of {col}')
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output / f'{col}_comparison.png', dpi=300)
    plt.show()

################################################################################
## estimating r, g, and s correlations using rolling regression
################################################################################
df = r_g_master.copy()
df.rename(columns = {
    'r (BEA / Jared)': 'r',
    'g (BEA / Jared)': 'g',
}, inplace=True)
df = df[['date', 'r', 'g']]

# now bring in deficit 
deficit_data = pd.read_excel(raw_data / 'r_g_historic_data.xlsx', sheet_name='master')
# convert year into datetime, say january 1st
deficit_data['date'] = pd.to_datetime(deficit_data['year'].astype(str) + '-01-01')
deficit_filtered = deficit_data[['date', 'primary deficit (billions)']].copy()
gdp = get_fred_series('GDPC1', 'gdp')
deficit_filtered = pd.merge(deficit_filtered, gdp, on='date', how='left')
# convert to percentage of gdp
deficit_filtered['primary deficit (billions)'] = (
    deficit_filtered['primary deficit (billions)'] / deficit_filtered['gdp']
) * 100

# merge on df 
df = pd.merge(df, deficit_filtered, on='date', how='left')
df.rename(columns={'primary deficit (billions)': 's'}, inplace=True)
# now we have df with r, g, and s

# collapse on the average for each year 
df['date'] = df['date'].dt.to_period('Y').dt.to_timestamp()
df = df.groupby('date').mean().reset_index()
# now set the date as index
df.set_index('date', inplace=True)

filtered_recession_df = filtered_recession_df[filtered_recession_df['date'] <= '2010-01-01']

def rolling_beta(df, y_col, x_col, min_obs=15, alpha=0.05):
    """
    Expanding-window OLS of y on x, returning beta and a two-sided
    (1-alpha) confidence interval for every start-year window.

    Returns
    -------
    DataFrame indexed by start year with columns
        ['beta', 'ci_lo', 'ci_hi']
    """
    rows   = []
    years  = df.index.to_numpy()
    last   = years[-1]
    for start in years:
        window = df.loc[start:last, [y_col, x_col]].dropna()
        n      = len(window)
        if n < min_obs:
            continue
        y = window[y_col].to_numpy()
        X = sm.add_constant(window[x_col])
        res = sm.OLS(y, X).fit()
        beta = res.params[x_col]
        se   = res.bse[x_col]
        tval = stats.t.ppf(1 - alpha/2, df=n-2)   # n-2 d.f.
        rows.append(
            dict(start=start,
                 beta=beta,
                 ci_lo=beta - tval*se,
                 ci_hi=beta + tval*se)
        )
    out = pd.DataFrame(rows).set_index('start')
    out.index.name = 'T0'
    return out
pairs = [("r", "g"), ("r", "s"), ("g", "r"),
         ("g", "s"), ("s", "r"), ("s", "g")]

beta_paths = {
    f"{y}_on_{x}": rolling_beta(df, y, x)
    for y, x in pairs
}
beta_df = pd.concat({k: v['beta'] for k, v in beta_paths.items()}, axis=1)
beta_df.to_csv(clean_data / 'rolling_betas.csv')
fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
axes = axes.flatten()
for ax, (label, dat) in zip(axes, beta_paths.items()):
    ax.plot(dat.index, dat['beta'], lw=1.8, label='β̂')
    ax.fill_between(dat.index, dat['ci_lo'], dat['ci_hi'],
                    color=ax.lines[-1].get_color(), alpha=0.25, linewidth=0)
    ax.axhline(0, lw=0.7, ls='--')
    ax.set_title(label.replace('_on_', ' on '))
    ax.set_xlabel(r'Start year $T_0$')
    ax.set_ylabel(r'$\hat\beta(T_0)$')
    ax.grid(True, ls='--', alpha=0.5)
    add_recession_bars(ax, filtered_recession_df, shortened=False)
fig.tight_layout()
plt.savefig(output / 'rolling_betas.png', dpi=300)
plt.show()

################################################################################
## plot collected forecasts
################################################################################
df.reset_index(inplace=True)
filtered_historical = df[df['date'] < '2025-01-01'].copy()
filtered_historical = filtered_historical[filtered_historical['date'] >= '2020-01-01']
forecasts = pd.read_csv(clean_data / 'master_projections_cleaned.csv', parse_dates=['date'])
# restrict to 10 year forecasts, so before 2035 
forecasts = forecasts[forecasts['date'] < '2035-01-01']
# restrict to post-2025 
forecasts = forecasts[forecasts['date'] >= '2025-01-01']
# forward fill CBO basline 
forecasts['r (cbo baseline)'] = forecasts['r (cbo baseline)'].ffill()
# first, plot forecasts for r 
plt.figure(figsize=(12, 6))
# plot historical data for r 
plt.plot(filtered_historical['date'], filtered_historical['r'],
         label='Historical Data', color='black', linestyle='-', linewidth=1.5)
plt.plot(forecasts['date'], forecasts['r (cbo baseline)'],
         label='CBO Baseline', linestyle='-')
plt.plot(forecasts['date'], forecasts['r (moody\'s S1)'],
            label='Moody\'s S1', linestyle='-')
plt.plot(forecasts['date'], forecasts['r (moody\'s S2)'],
            label='Moody\'s S2', linestyle='-')
plt.plot(forecasts['date'], forecasts['r (moody\'s S3)'],
            label='Moody\'s S3', linestyle='-')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(pd.to_datetime('2025-01-01'), color='black', linestyle='--')
plt.legend()
plt.title('Forecasts of Real Interest Rate')
plt.ylabel('Real Interest Rate (%)')
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output / 'forecasts_r.png', dpi=300)
plt.show()

# now plot forecasts for g
forecasts['g (cbo baseline)'] = forecasts['g (cbo baseline)'].ffill()
plt.figure(figsize=(12, 6))
# plot historical data for g
plt.plot(filtered_historical['date'], filtered_historical['g'],
         label='Historical Data', color='black', linestyle='-', linewidth=1.5)
plt.plot(forecasts['date'], forecasts['g (cbo baseline)'],
         label='CBO Baseline', linestyle='-')
plt.plot(forecasts['date'], forecasts['g (moody\'s S1)'],
            label='Moody\'s S1', linestyle='-')
plt.plot(forecasts['date'], forecasts['g (moody\'s S2)'],
            label='Moody\'s S2', linestyle='-')
plt.plot(forecasts['date'], forecasts['g (moody\'s S3)'],
            label='Moody\'s S3', linestyle='-')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(pd.to_datetime('2025-01-01'), color='black', linestyle='--')
plt.legend()
plt.title('Forecasts of Real GDP Growth Rate')
plt.ylabel('Real GDP Growth Rate (%)')
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output / 'forecasts_g.png', dpi=300)
plt.show()

# now plot forecasts for s 
forecasts['s (cbo baseline)'] = forecasts['s (cbo baseline)'].ffill()
forecasts['s (tbl senate, as written)'] = forecasts['s (tbl senate, as written)'].ffill()
forecasts['s (tbl senate, permanent)'] = forecasts['s (tbl senate, permanent)'].ffill()
# first, plot historical data for s
plt.figure(figsize=(12, 6))
# plot historical data for s
plt.plot(filtered_historical['date'], filtered_historical['s'],
         label='Historical Data', color='black', linestyle='-', linewidth=1.5)
plt.plot(forecasts['date'], forecasts['s (cbo baseline)'],
         label='CBO Baseline', linestyle='-')
plt.plot(forecasts['date'], forecasts['s (tbl senate, permanent)'],
            label='Senate TBL Permanent', linestyle='-')
plt.plot(forecasts['date'], forecasts['s (tbl senate, as written)'],
            label='Senate TBL As Written', linestyle='-')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(pd.to_datetime('2025-01-01'), color='black', linestyle='--')
plt.legend()
plt.title('Forecasts of Primary Deficit as % of GDP')
plt.ylabel('Primary Deficit as % of GDP')
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output / 'forecasts_s.png', dpi=300)
plt.show()

################################################################
# figure: illustrative debt dynamics: r < g vs r > g
################################################################
# Parameters
n_years = 30
b0 = 1.0  # starting debt as a percentage of GDP for year = 1
# Scenarios
scenarios = {
    "r > g (deficit)": {"r": 0.04, "g": 0.02, 'd': -0.02},
    "r < g (deficit)": {"r": 0.02, "g": 0.03, 'd': -0.02},
    "r > g (surplus)": {"r": 0.04, "g": 0.02, 'd': 0.02},
    "r < g (surplus)": {"r": 0.02, "g": 0.03, 'd': 0.02},
}
results = {}
table_rows = []
for i, (label, params) in enumerate(scenarios.items(), start=1):
    r, g, d = params["r"], params["g"], params["d"]
    t = np.arange(n_years)
    growth_factor = ((1 + r) / (1 + g)) ** t
    b_path = b0 * growth_factor - d * (1 - growth_factor) / (1 - (1 + r)/(1 + g))
    b_path *= 100  # Convert to percent of GDP
    # Store results: level, slope (% change), second derivative
    level = b_path
    slope = np.empty_like(level)
    slope[1:] = 100 * (level[1:] - level[:-1]) / level[:-1]
    slope[0] = np.nan  # undefined
    curvature = np.empty_like(level)
    curvature[2:] = 100 * (slope[2:] - slope[1:-1]) / np.abs(slope[1:-1])
    curvature[:2] = np.nan  # undefined
    results[label] = {
        "level": level,
        "slope": slope,
        "curvature": curvature
    }
    table_rows.append([f"({i})", f"{r:.0%}", f"{g:.0%}", f"{d:.0%}"])
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
titles = [
    "Debt Level (% of GDP)",
    "Slope (% change from prior year)",
    "Curvature (% change in slope)"
]
for i, (label, metrics) in enumerate(results.items(), start=1):
    color = f"C{i}"
    axes[0].plot(range(n_years), metrics["level"], label=f"({i}) {label}", color=color)
    axes[1].plot(range(n_years), metrics["slope"], color=color)
    axes[2].plot(range(n_years), metrics["curvature"], color=color)

axes[2].set_xlabel("Years")
axes[0].set_ylabel("Level")
axes[1].set_ylabel("Slope")
axes[2].set_ylabel("Curvature")
axes[0].axhline(100, color='gray', linestyle='--')
axes[0].legend(loc='upper left', fontsize=9)
'''fig.suptitle("Debt Dynamics w/Static Variables – Level, Slope, Curvature", 
             fontsize=14, weight='bold')'''
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(output / "illustrative_debt_dynamics_1.png")
plt.show()

# now plot same figure, with an added twist: 
# now there is a feedback loop:
# 1 percent increase in the debt ratio leads to the interest rates increasing by 2.5 basis points
alpha = 0.00025   # 2.5 basis points per 1ppt increase in debt-to-GDP
b0_ref = 100      # reference point for interest rate feedback

results = {}
table_rows = []

for i, (label, params) in enumerate(scenarios.items(), start=1):
    r0, g, d = params["r"], params["g"], params["d"]
    b_path = np.zeros(n_years)
    b_path[0] = b0 * 100  # express in % of GDP

    for t in range(1, n_years):
        # Interest rate with feedback based on prior year's debt
        r_t = r0 + alpha * (b_path[t - 1] - b0_ref)

        # Update debt using recursive formula
        b_path[t] = ((1 + r_t) / (1 + g)) * b_path[t - 1] - d * 100

    # Compute slope and curvature
    slope = np.empty_like(b_path)
    slope[1:] = 100 * (b_path[1:] - b_path[:-1]) / b_path[:-1]
    slope[0] = np.nan

    curvature = np.empty_like(b_path)
    curvature[2:] = 100 * (slope[2:] - slope[1:-1]) / np.abs(slope[1:-1])
    curvature[:2] = np.nan

    results[label] = {
        "level": b_path,
        "slope": slope,
        "curvature": curvature
    }

    table_rows.append([f"({i})", f"{r0:.0%}", f"{g:.0%}", f"{d:.0%}"])
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
titles = [
    "Debt Level (% of GDP)",
    "Slope (% change from prior year)",
    "Curvature (% change in slope)"
]
for i, (label, metrics) in enumerate(results.items(), start=1):
    color = f"C{i}"
    axes[0].plot(range(n_years), metrics["level"], label=f"({i}) {label}", color=color)
    axes[1].plot(range(n_years), metrics["slope"], color=color)
    axes[2].plot(range(n_years), metrics["curvature"], color=color)

axes[2].set_xlabel("Years")
axes[0].set_ylabel("Level")
axes[1].set_ylabel("Slope")
axes[2].set_ylabel("Curvature")
axes[0].axhline(100, color='gray', linestyle='--')
axes[0].legend(loc='upper left', fontsize=9)
'''fig.suptitle("Debt Dynamics w/Static Variables – Level, Slope, Curvature w/Interest Rate Feedback", 
             fontsize=14, weight='bold')'''
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(output / "illustrative_debt_dynamics_2.png")
plt.show()

################################################################################
## estimate interest rate equation
################################################################################

# first, bring in 10-year treasury term premium
term_prem = pd.read_csv(raw_data / 'acm_term_premium.csv')
term_prem.columns = term_prem.columns.str.lower()
term_prem = term_prem[['date', 'acmtp10']]
term_prem['date'] = pd.to_datetime(term_prem['date'])
# convert to quarterly average 
term_prem['date'] = term_prem['date'].dt.to_period('Q').dt.to_timestamp()
term_prem = term_prem.groupby('date').mean().reset_index()
# print maximum and minimum dates 
print(f"Term premium data from {term_prem['date'].min()} to {term_prem['date'].max()}")

rgdp = get_fred_series('GDPC1', 'real_gdp')
print(f"Real GDP data from {rgdp['date'].min()} to {rgdp['date'].max()}")
potential_gdp = get_fred_series('GDPPOT', 'potential_gdp')
print(f"Potential GDP data from {potential_gdp['date'].min()} to {potential_gdp['date'].max()}")
debt = get_fred_series('FYGFGDQ188S', 'debt_gdp')
print(f"Debt-to-GDP data from {debt['date'].min()} to {debt['date'].max()}")
# nominal GDP for primary surplus calculation
ngdp = get_fred_series('GDP', 'nominal_gdp')
ngdp['nominal_gdp'] = ngdp['nominal_gdp'] * 1000
print(f"Nominal GDP data from {ngdp['date'].min()} to {ngdp['date'].max()}")

# bring in primary surplus data (bea)
primary = pd.read_csv(raw_data / 'primary_surplus_bea.csv')
# create datetime variable; right now, quarter looks like: '1947q1'
quarter_map = {
    'q1': '-01-01',
    'q2': '-04-01',
    'q3': '-07-01',
    'q4': '-10-01'
}
primary['date'] = primary['quarter'].str.lower().replace(quarter_map, regex=True)
primary['date'] = pd.to_datetime(primary['date'])
primary = primary[['date', 'primary_surplus']]
# drop missing primary_surplus 
primary['primary_surplus'] = primary['primary_surplus'].astype(str).str.replace(',', '')
primary['primary_surplus'] = pd.to_numeric(primary['primary_surplus'], errors='coerce')
# drop missing 
primary = primary.dropna(subset=['primary_surplus'])
print(f"Primary surplus data from {primary['date'].min()} to {primary['date'].max()}")

df = pd.merge(term_prem, rgdp, on='date', how='left')
df = pd.merge(df, ngdp, on='date', how='left')
df = pd.merge(df, potential_gdp, on='date', how='left')
df = pd.merge(df, debt, on='date', how='left')
df = pd.merge(df, primary, on='date', how='left')

# convert all vars to float 
for var in ['acmtp10', 'real_gdp', 'nominal_gdp', 'potential_gdp', 'debt_gdp', 'primary_surplus']:
    # first, strip all ','
    df[var] = df[var].astype(str).str.replace(',', '')
    df[var] = df[var].astype(float)

df['output_gap'] = ((df['real_gdp'] - df['potential_gdp']) / df['potential_gdp']) * 100.0
df['primary_def_gdp'] = (df['primary_surplus'] / df['nominal_gdp']) * 100.0

df.rename(
    columns = {
        'acmtp10': 'ACMTP10_Q',
        'primary_def_gdp': 'primary_bal_gdp'
    }, inplace=True
)

required = ['ACMTP10_Q', 'debt_gdp', 'primary_bal_gdp', 'output_gap']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Merged DF missing columns: {missing}")
# drop rows with missing values in required columns
df = df.dropna(subset=required)

d = df.copy()
d['b_lag']       = d['debt_gdp'].shift(1)
d['y_lag']       = d['ACMTP10_Q'].shift(1)
d['primary_lag'] = d['primary_bal_gdp'].shift(1)

# Keep date and set it as index
base = d[['date','ACMTP10_Q','b_lag','output_gap','primary_lag','y_lag']].dropna().copy()
base['date'] = pd.to_datetime(base['date'])
# optional: align to quarter **end** so your cut dates like 2007-12-31 behave exactly as expected
base['date'] = base['date'].dt.to_period('Q').dt.to_timestamp(how='end')
base = base.set_index('date').sort_index()

# Subsamples
POST_GFC_START = pd.Timestamp('2009-01-01')   # 2009Q1+
PRE_GFC_END    = pd.Timestamp('2007-12-31')   # through 2007Q4

base_fd = base.copy()

# First difference the main variables
base_fd['d_acm_tp'] = base_fd['ACMTP10_Q'].diff()
base_fd['d_debt'] = base_fd['b_lag'].diff()
base_fd['d_output_gap'] = base_fd['output_gap'].diff()
base_fd['d_primary'] = base_fd['primary_lag'].diff()

# Keep lagged level of term premium for AR component
base_fd['y_lag'] = base_fd['y_lag']  # This stays in levels

# Drop first observation (lost to differencing)
base_fd = base_fd.dropna()

def fit_hac_fd(df, hac_lags=4):
    """First differences specification"""
    y = df['d_acm_tp']  # Change in term premium
    X = sm.add_constant(df[['d_debt', 'd_output_gap', 'd_primary', 'y_lag']])
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})

# Run first differences regressions
full_fd = base_fd
post_fd = base_fd.loc[base_fd.index >= POST_GFC_START]
pre_fd = base_fd.loc[base_fd.index <= PRE_GFC_END]

res_full_fd = fit_hac_fd(full_fd)
res_post_fd = fit_hac_fd(post_fd)
res_pre_fd = fit_hac_fd(pre_fd)

# Convert to basis points
beta_r_full_fd_bps = res_full_fd.params['d_debt'] * 100.0
beta_r_post_fd_bps = res_post_fd.params['d_debt'] * 100.0
beta_r_pre_fd_bps = res_pre_fd.params['d_debt'] * 100.0

print("=== RESULTS WITH FIRST DIFFERENCES ===")
print(f"β_r (bps per 1-pp change in debt): Full={beta_r_full_fd_bps:.2f}, Post-GFC={beta_r_post_fd_bps:.2f}, Pre-GFC={beta_r_pre_fd_bps:.2f}")

# Create table
info_fd = {
    'N': lambda m: f"{int(m.nobs)}",
    'R$^2$': lambda m: f"{m.rsquared:.3f}",
    'SE': lambda m: m.cov_type
}

tbl_fd = summary_col(
    results=[res_full_fd, res_post_fd, res_pre_fd],
    float_format='%0.3f',
    stars=True,
    model_names=['Full sample', 'Post-GFC', 'Pre-GFC'],
    info_dict=info_fd,
    regressor_order=['const', 'd_debt', 'd_output_gap', 'd_primary', 'y_lag']
)

tbl_fd.add_title('ACM 10y Term Premium: First Differences Specification (HAC SEs)')
latex_str_fd = tbl_fd.as_latex()
print(latex_str_fd)

with open(f'{output}/acm_tp_regressions_first_diff.tex', 'w') as f:
    f.write(latex_str_fd)