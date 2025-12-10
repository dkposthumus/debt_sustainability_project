import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fredapi import Fred
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
import scipy.stats as st

# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'debt_sustainability_project')
data = (work_dir / 'data')
raw_data = (data / 'sdsa' / 'raw')
output = (work_dir / 'output' / 'sdsa' / 'graphics' / 'other')
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
# Plot Natural Interest Rate
################################################################################
natural_interest_rate = pd.read_excel(raw_data / 'natural_interest_rate.xlsx', 
                                      sheet_name='HLW Estimates', skiprows=5)
natural_interest_rate.columns = natural_interest_rate.columns.str.lower()
natural_interest_rate.rename(
    columns = {
        'us.2': 'us_natural_interest rate'
    }, inplace=True
)
natural_interest_rate['date'] = pd.to_datetime(natural_interest_rate['date'])
# plot annual interest rate, the full historical series
plt.figure(figsize=(12, 8))
plt.plot(natural_interest_rate['date'], 
         natural_interest_rate['us_natural_interest rate'],
         label='Natural Interest Rate (HLW Model)', color='blue', alpha=0.75)
# add recession bars
add_recession_bars(plt.gca(), filtered_recession_df)
plt.title('Natural Interest Rate in the US (1962-2023)')
plt.ylabel('Natural Interest Rate (%)')
plt.grid(True)
plt.legend(fontsize='x-large', loc='best')
plt.tight_layout()
plt.savefig(output / 'natural_interest_rate_full.pdf', dpi=300)
plt.show()

natural_interest_rate = natural_interest_rate[natural_interest_rate['date'] >= '2004-01-01']
plt.figure(figsize=(12, 8))
plt.plot(natural_interest_rate['date'], 
         natural_interest_rate['us_natural_interest rate'],
         label='Natural Interest Rate (HLW Model)', color='blue', alpha=0.75)
# add horizontal line at post-GFC average, with text label
post_gfc_start = pd.to_datetime('2009-01-01')
post_gfc_data = natural_interest_rate[natural_interest_rate['date'] >= post_gfc_start]
average_post_gfc_rate = post_gfc_data['us_natural_interest rate'].mean()
plt.axhline(average_post_gfc_rate, color='green', linestyle='--', 
            label='Post-GFC Average Rate: {:.2f}%'.format(average_post_gfc_rate))
plt.text(pd.to_datetime('2013-06-01'), 
         average_post_gfc_rate + 0.15, 
         'Post-GFC Average Rate: {:.2f}%'.format(average_post_gfc_rate),
         color='green', fontsize=25, ha='center', va='bottom')
plt.title('Natural Interest Rate in the US')
plt.ylabel('Natural Interest Rate (%)')
plt.grid(True)
shortened = recession_df[recession_df['date'] >= '2004-01-01']
add_recession_bars(plt.gca(), shortened)
plt.legend(fontsize='x-large', loc='best')
plt.tight_layout()
plt.savefig(output / 'natural_interest_rate.pdf', dpi=300)
plt.show()

################################################################################
# Historical Curvature of the Debt Path
################################################################################
def curvature_savgol(b_series, window_length=9, polyorder=2, use_log=False):
    """
    b_series: 1D array-like (level, e.g., debt/GDP in decimals)
    window_length: odd int >= 3
    polyorder: typically 2 or 3
    use_log: if True, apply SG to log(b); then slope≈growth rate, curvature≈change in growth
    """
    x = np.asarray(b_series, dtype=float)
    if use_log:
        if np.any(x <= 0):
            raise ValueError("use_log=True requires strictly positive series.")
        x = np.log(x)

    # enforce odd window <= len(x) and > polyorder
    w = window_length if window_length % 2 == 1 else window_length + 1
    w = min(w, len(x) - 1 if (len(x) % 2 == 0) else len(x))  # largest odd ≤ len(x)
    if w < 3: 
        raise ValueError("Series too short for Savitzky–Golay smoothing.")
    if w <= polyorder:
        polyorder = max(1, min(polyorder, w - 1))

    slope = savgol_filter(x, w, polyorder, deriv=1, delta=1.0, mode='interp')
    curv  = savgol_filter(x, w, polyorder, deriv=2, delta=1.0, mode='interp')
    return slope, curv

debt = get_fred_series('FYGFGDQ188S', 'Federal Debt Held by the Public')
debt.columns = debt.columns.str.lower()
debt.rename(columns={'federal debt held by the public': 'debt'}, inplace=True)
debt['debt'] = debt['debt'] / 100.0  # percent → decimal

# keep one observation per year (Q4). Many FRED quarter stamps are quarter-start dates; Q4==October works.
debt_q4 = debt[debt['date'].dt.month == 10].sort_values('date').copy()

# --- 3) Apply Savitzky–Golay derivatives ---
# (A) Curvature of the *level* series (units: “per year^2” of debt/GDP)
slope_sg, curv_sg = curvature_savgol(debt_q4['debt'], window_length=3, polyorder=2, use_log=False)
debt_q4['slope_sg'] = slope_sg
debt_q4['curvature_sg'] = curv_sg

# (B) Curvature of *log(debt)* (scale-free; think growth-rate and change-in-growth)
slope_log_sg, curv_log_sg = curvature_savgol(debt_q4['debt'], window_length=9, polyorder=2, use_log=True)
debt_q4['slope_log_sg'] = slope_log_sg         # ≈ d/dt ln b_t  (annual growth rate of debt/GDP)
debt_q4['curvature_log_sg'] = curv_log_sg      # ≈ d^2/dt^2 ln b_t (change in growth)

# plot smoothed and observed series together
plt.figure(figsize=(12, 8))
plt.plot(debt_q4['date'], debt_q4['debt'], label='Observed Debt/GDP (Q4)', color='black', alpha=0.5)
plt.plot(debt_q4['date'], debt_q4['slope_sg'], label='Slope (SG, W=9, p=2)', color='blue', alpha=0.75)
plt.plot(debt_q4['date'], debt_q4['curvature_sg'], label='Curvature (SG, W=9, p=2)', color='red', alpha=0.75)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('')
plt.xlabel('')
plt.ylabel('Debt/GDP (level) and derivatives')
plt.grid(True)
filtered_recession_df = recession_df[recession_df['date'] >= debt_q4['date'].min()]
add_recession_bars(plt.gca(), filtered_recession_df)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(output / 'debt_savgol.pdf', dpi=300)
plt.show()

# --- 4) Plot (pick one of these two “curvatures” and use it everywhere, incl. your sims) ---
plt.figure(figsize=(12, 8))
plt.plot(debt_q4['date'], debt_q4['curvature_log_sg'], label='Curvature of log(debt/GDP) (SG, W=9, p=2)', alpha=0.85)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('')
plt.xlabel('')
plt.ylabel('Second derivative (per year²)')
plt.grid(True)
filtered_recession_df = recession_df[recession_df['date'] >= debt_q4['date'].min()]
add_recession_bars(plt.gca(), filtered_recession_df)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(output / 'debt_curvature_savgol_logQ4.pdf', dpi=300)
plt.show()

################################################################################
# compare distributions between historical data and simulation results
################################################################################
output = (work_dir / 'output' / 'sdsa' / 'graphics')

sim_results = pd.read_csv(f'{output}/sdsa_enrichment1_sim_results.csv')
# now we want to create separate vectors of the 10-year change (in percentage points)
# in debt/GDP for both the historical data and the simulation results
debt_q4['debt_10yr_change'] = debt_q4['debt'].diff(10) * 100  # convert to percentage points
# drop NA values
historical_changes = debt_q4['debt_10yr_change'].dropna()

# # plot the distributions using seaborn
plt.figure(figsize=(12, 8))
ax = plt.gca()
# (Optional but recommended) consistent binning across all overlays
all_changes = []
# ---- simulated distributions (step outlines, no fill) ----
for c_vals, color in zip([0, 0.15, 0.30], ['blue', 'green', 'red']):
    subset = sim_results[sim_results['c'] == c_vals]
    yr10_changes = []
    for i in subset['sim'].unique():
        sim_df = subset[subset['sim'] == i].sort_values('year').reset_index(drop=True)
        change = (sim_df['b'].iloc[-1] - sim_df['b'].iloc[0]) * 100  # percentage points
        yr10_changes.append(change)
    all_changes.extend(yr10_changes)
# include historical in bin range too
all_changes.extend(list(historical_changes))
# define shared bins (step-look is much nicer with shared bins)
bins = np.histogram_bin_edges(all_changes, bins=30)
for c_vals, color in zip([0, 0.15, 0.30], ['blue', 'green', 'red']):
    subset = sim_results[sim_results['c'] == c_vals]
    yr10_changes = []
    for i in subset['sim'].unique():
        sim_df = subset[subset['sim'] == i].sort_values('year').reset_index(drop=True)
        change = (sim_df['b'].iloc[-1] - sim_df['b'].iloc[0]) * 100
        yr10_changes.append(change)
    sns.histplot(
        yr10_changes,
        bins=50,
        stat="density",
        element="step",
        fill=False,
        linewidth=1.6,
        color=color,
        label=f"Simulated 10-year Changes (c={c_vals})",
        ax=ax,
    )
    # median line
    median_change = np.median(yr10_changes)
    ax.axvline(
        median_change,
        linestyle="--",
        color=color,
        linewidth=1.6,
        label=f"_nolegend_",
    )
# ---- observed / historical distribution (bold black step outline) ----
sns.histplot(
    historical_changes,
    bins=20,
    stat="density",
    element="step",
    fill=False,
    linewidth=3.0,
    color="black",
    label="Observed 10-year Changes",
    ax=ax,
)
median_historical = np.median(historical_changes)
ax.axvline(
    median_historical,
    color="black",
    linestyle="--",
    linewidth=3.0,
    label="_nolegend_",
)
ax.set_title("")
ax.set_xlabel("10-year Change in Debt/GDP (percentage points)")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{output}/debt_10yr_change_distribution.pdf", dpi=300)
plt.show()

################################################################################
# quantile tables
################################################################################
# --- Gather distributions ---
distributions = {}
for c_vals in [0, 0.15, 0.30]:
    subset = sim_results[sim_results['c'] == c_vals]
    yr10_changes = []
    for i in subset['sim'].unique():
        sim_df = subset[subset['sim'] == i].sort_values('year').reset_index(drop=True)
        change = (sim_df['b'].iloc[-1] - sim_df['b'].iloc[0]) * 100  # percentage points
        yr10_changes.append(change)
    distributions[f'$c={c_vals:.2f}$'] = np.array(yr10_changes)

distributions['Historical'] = np.array(historical_changes)

# --- Helper: empirical CDF percentile ranks ---
def empirical_percentile(x, values):
    """Compute empirical percentile rank of each value in `values` within sample x."""
    return np.array([st.percentileofscore(x, v, kind='weak') for v in values])

# --------------------------------------------------------------------------
# 1️⃣ Table: percentile ranks of simulated distributions within historical
# --------------------------------------------------------------------------
summary_rows = []
hist = distributions['Historical']

for key, vals in distributions.items():
    if key == 'Historical':
        continue
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    hist_perc = empirical_percentile(hist, [p10, p50, p90])
    summary_rows.append({
        'Simulation': key,
        '10th Percentile (pp)': p10,
        'Rank in Hist. (\\%)_1': hist_perc[0],
        'Median (pp)': p50,
        'Rank in Hist. (\\%)_2': hist_perc[1],
        '90th Percentile (pp)': p90,
        'Rank in Hist. (\\%)_3': hist_perc[2],
    })

table1 = pd.DataFrame(summary_rows)

# Drop the duplicate suffixes and standardize names
table1.columns = [
    'Simulation',
    '10th Percentile (pp)',
    'Rank in Hist. (\\%)',
    'Median (pp)',
    'Rank in Hist. (\\%)',
    '90th Percentile (pp)',
    'Rank in Hist. (\\%)'
]

# --- Force numeric values to one-decimal-place strings safely ---
def format_one_decimal(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return x

table1 = table1.applymap(format_one_decimal)

# --- Export LaTeX ---
table1_latex = table1.to_latex(
    index=False,
    escape=False,
    column_format='lcccccc',
    caption='Percentile ranks of simulated 10-year debt/GDP changes within the historical distribution.',
    label='tab:sim_vs_hist_percentile_ranks',
    bold_rows=False,
    longtable=False
)

with open(output / 'table_sim_vs_hist_percentile_ranks.tex', 'w') as f:
    f.write(table1_latex)

# --------------------------------------------------------------------------
# 2️⃣ Table: empirical percentiles of each distribution
# --------------------------------------------------------------------------
quantiles = [10, 25, 50, 75, 90]
table2 = pd.DataFrame({
    dist_name: np.percentile(values, quantiles)
    for dist_name, values in distributions.items()
}, index=[f'{q}th' for q in quantiles])

table2 = table2.T.reset_index().rename(columns={'index': 'Distribution'})
table2 = table2.applymap(format_one_decimal)

table2_latex = table2.to_latex(
    index=False,
    escape=False,
    column_format='lccccc',
    caption='Empirical percentiles (10th–90th) of 10-year debt/GDP changes: historical and simulated distributions.',
    label='tab:debt_change_percentiles',
    bold_rows=False,
    longtable=False
)

with open(output / 'table_debt_change_percentiles.tex', 'w') as f:
    f.write(table2_latex)

print("\n✅ LaTeX tables saved to:")
print(f" - {output / 'table_sim_vs_hist_percentile_ranks.tex'}")
print(f" - {output / 'table_debt_change_percentiles.tex'}")