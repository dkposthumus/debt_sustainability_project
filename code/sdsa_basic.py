import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from fredapi import Fred
from pathlib import Path
import statsmodels.api as sm

home = Path.home()
work_dir = (home / 'debt_sustainability_project')
data = (work_dir / 'data' / 'sdsa')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
output = (work_dir / 'output' / 'sdsa' / 'graphics')
code = Path.cwd() 

# set matplotlib style
plt.style.use('mahoney_lab.mplstyle')

################################################################################
# 1. Define relevant functions
################################################################################
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
fred = Fred(api_key='8905b2f5faefd705486e644f09bb8088')

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

# prep recession data
recession = get_fred_series('USRECD', 'Recession')
recession['recession'] = recession['Recession'].astype(int)
recession['date'] = pd.to_datetime(recession['date'])
recession_df = recession[['date', 'recession']].copy()
recession_df['date'] = pd.to_datetime(recession_df['date'])
recession_df['recession'] = recession_df['recession'].astype(int)

################################################################################
## 2. Calibrate / set parameters - random walk
################################################################################
graphics_path = output / 'sdsa_random_walk'
x0 = 0 
s_x = 0.003
b0 = 1.21
a_u = -0.03
a_u = 0
s_u = 0.01
a_s = -0.02
s_s = 0.01

n_years = 10
n_simulations = 5000

c_vals_to_run = [0, 0.33, 0.67, 1.0]

# Set random seed for reproducibility
np.random.seed(42)

# Container for results
results = []

for c in c_vals_to_run:
    for sim in range(n_simulations):
        x = np.zeros(n_years)
        u = np.zeros(n_years)
        s = np.zeros(n_years)
        b = np.zeros(n_years)
        rg = np.zeros(n_years)
        x[0] = x0
        b[0] = b0
        u[0] = a_u + np.random.normal(0, s_u)
        rg[0] = x[0] + u[0]
        s[0] = (1 - c) * a_s + c * (rg[0] * b[0]) + np.random.normal(0, s_s)

        # Initialize the first year
        x[1] = x[0] + np.random.normal(0, s_x)
        u[1] = a_u + np.random.normal(0, s_u)
        rg[1] = x[1] + u[1]
        s[1] = (1 - c) * a_s + c * (rg[1] * b[0]) + np.random.normal(0, s_s)
        b[1] = b[0] + (rg[1] * b[0]) - s[1]
        for t in range(2, n_years):
            e_x = np.random.normal(0, s_x)
            e_u = np.random.normal(0, s_u)
            e_s = np.random.normal(0, s_s)
            x[t] = x[t - 1] + e_x
            u[t] = a_u + e_u
            s[t] = (1 - c) * a_s + c * ((x[t] + u[t]) * b[t - 1]) + e_s
            b[t] = b[t - 1] + (x[t] + u[t]) * b[t - 1] - s[t]
        for t in range(n_years):
            results.append({
                "year": t + 1,
                "b": b[t],
                "x": x[t],
                "u": u[t],
                "s": s[t],
                "sim": sim,
                "c": c
            })

df_sdsa = pd.DataFrame(results)

# now we want to compile the 10-year change in debt for each simulation
df_sdsa['b_change'] = df_sdsa.groupby(['sim', 'c'])['b'].transform(lambda x: x.iloc[-1] - x.iloc[0])
# now plot the distributions for each c value 
plt.figure(figsize=(12, 8))
plt.hist(df_sdsa[df_sdsa['c'] == 0]['b_change'], bins=30, alpha=0.5, label='c = 0')
plt.hist(df_sdsa[df_sdsa['c'] == 0.33]['b_change'], bins=30, alpha=0.5, label='c = 0.33')
plt.hist(df_sdsa[df_sdsa['c'] == 0.67]['b_change'], bins=30, alpha=0.5, label='c = 0.67')
plt.hist(df_sdsa[df_sdsa['c'] == 1.0]['b_change'], bins=30, alpha=0.5, label='c = 1.0')
plt.xlabel('10-Year Change in Debt (b)')
plt.ylabel('Frequency')
plt.title('Distribution of 10-Year Change in Debt by c Value')
plt.legend(loc='best', fontsize='x-large')
plt.grid()
plt.tight_layout()
plt.savefig(f'{graphics_path}/sdsa_debt_random_walk_distributions.png', dpi=300)
plt.show()

# now collapse on the 25th, median, and 75th percentiles for each c value
df_summary = df_sdsa.groupby(['c', 'year']).agg(
    b_25th=('b', lambda x: np.percentile(x, 25)),
    b_median=('b', 'median'),
    b_75th=('b', lambda x: np.percentile(x, 75))
).reset_index()
plt.figure(figsize=(12, 8))
for c in c_vals_to_run:
    subset = df_summary[df_summary['c'] == c]
    plt.plot(subset['year'], subset['b_median'], label=f'c = {c}')
    plt.fill_between(subset['year'], subset['b_25th'], subset['b_75th'], alpha=0.2)
# plot initial debt level
plt.axhline(y=b0, color='black', linestyle='--', label=f'Initial Debt Level ({b0})')
plt.xlabel('Year')
plt.ylabel('Debt Level')
plt.legend(loc='best', fontsize='x-large')
plt.title('Median and Percentiles of Debt Level by c Value')
plt.grid()
plt.tight_layout()
plt.savefig(f'{graphics_path}/sdsa_debt_random_walk_path.png', dpi=300)
plt.show()
# save and export df_sdsa 
df_sdsa.to_csv(clean_data / 'sdsa_random_walk_results.csv', index=False)

c = 0.5  # feedback strength fixed

# Scenarios: (x0, a_u)
scenarios = {
    "(1)": (-0.02, 0),
    "(2)": (0, -0.02),
    "(3)": (-0.02, -0.02),
    "(4)": (0.02, 0),
    "(5)": (0, 0.02),
    "(6)": (0.02, 0.02),
}
colors = {
    "(1)": "orange",
    "(2)": "green",
    "(3)": "pink",
    "(4)": "brown",
    "(5)": "purple",
    "(6)": "gold",
}

# Run simulations
results = []
for label, (x0, a_u) in scenarios.items():
    for sim in range(n_simulations):
        x = np.zeros(n_years)
        u = np.zeros(n_years)
        s = np.zeros(n_years)
        b = np.zeros(n_years)
        rg = np.zeros(n_years)
        x[0] = x0
        b[0] = b0
        u[0] = a_u + np.random.normal(0, s_u)
        rg[0] = x[0] + u[0]
        s[0] = (1 - c) * a_s + c * (rg[0] * b[0]) + np.random.normal(0, s_s)

        for t in range(1, n_years):
            x[t] = x[t-1] + np.random.normal(0, s_x)
            u[t] = a_u + np.random.normal(0, s_u)
            rg[t] = x[t] + u[t]
            s[t] = (1 - c) * a_s + c * (rg[t] * b[t-1]) + np.random.normal(0, s_s)
            b[t] = b[t-1] + rg[t] * b[t-1] - s[t]
        
        for t in range(n_years):
            results.append({
                "year": t+1,
                "rg": rg[t],
                "b": b[t],
                "scenario": label,
                "sim": sim
            })

df = pd.DataFrame(results)

# Compute means
df_mean = df.groupby(['year', 'scenario'])[['rg', 'b']].mean().reset_index()

# Plot mean r - g
plt.figure(figsize=(10, 6))
for label in scenarios.keys():
    subset = df_mean[df_mean['scenario'] == label]
    plt.plot(subset['year'], subset['rg'], label=label, color=colors[label])
plt.title("Mean Interest-Growth Differential (r - g)")
plt.xlabel("Year")
plt.ylabel("r - g")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{graphics_path}/sdsa_random_walk_mean_rg.png', dpi=300)
plt.show()

# Plot mean debt path
plt.figure(figsize=(10,6))
for label in scenarios.keys():
    subset = df_mean[df_mean['scenario'] == label]
    plt.plot(subset['year'], subset['b'], label=label, color=colors[label])
plt.axhline(y=b0, color='black', linestyle='--', label=f'Initial Debt Level ({b0})')
plt.title("Mean Debt-to-GDP Ratio")
plt.xlabel("Year")
plt.ylabel("Debt (as % of GDP)")
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{graphics_path}/sdsa_random_walk_rg_varying_debt.png', dpi=300)
plt.show()

################################################################################
## 3. Calibrate / set parameters - better / more specific parameters
################################################################################
graphics_path = output / 'sdsa_basic'
# pull in historical data for r, g, and s
gdp = get_fred_series('A191RL1A225NBEA', 'GDP')
gdp.dropna(inplace=True)
# extract year and collapse average on annual
gdp['year'] = gdp['date'].dt.year
gdp['g'] = gdp['GDP']
gdp = gdp[['year', 'g']]

interest_rates = get_fred_series('DGS10', 'Interest Rate (10-Year Treasury)')
interest_rates['r'] = interest_rates['Interest Rate (10-Year Treasury)']
interest_rates.dropna(inplace=True)
# extract year and collapse average on annual
interest_rates['year'] = interest_rates['date'].dt.year
interest_rates = interest_rates.groupby('year')['r'].mean().reset_index()
interest_rates = interest_rates[['year', 'r']]

inflation = get_fred_series('EXPINF10YR', 'Inflation (10-Year Expected)')
inflation.dropna(inplace=True)
# extract year and collapse average on annual
inflation['year'] = inflation['date'].dt.year
inflation = inflation.groupby('year')['Inflation (10-Year Expected)'].mean().reset_index()
inflation = inflation[['year', 'Inflation (10-Year Expected)']]

deficit = get_fred_series('FYFSGDA188S', 'Federal Deficit')
deficit.dropna(inplace=True)
# extract year and collapse average on annual
deficit['year'] = deficit['date'].dt.year
deficit = deficit.groupby('year')['Federal Deficit'].mean().reset_index()
deficit = deficit[['year', 'Federal Deficit']]
interest_outlays = get_fred_series('FYOIGDA188S', 'Interest Outlays')
interest_outlays.dropna(inplace=True)
# extract year and collapse average on annual
interest_outlays['year'] = interest_outlays['date'].dt.year
interest_outlays = interest_outlays.groupby('year')['Interest Outlays'].mean().reset_index()
interest_outlays = interest_outlays[['year', 'Interest Outlays']]

# bring in smoothed GDP data
gdp_smooth = pd.read_excel(raw_data / 'rl_gdp_smooth_jb.xlsx', skiprows=4)
# collapse on year, which i extract from 'unnamed: 5' colum (first four characters)
gdp_smooth['year'] = gdp_smooth['Unnamed: 5'].str[:4].astype(int)
gdp_smooth = gdp_smooth.groupby('year')['HP1_INFO'].mean().reset_index()
gdp_smooth.rename(columns={'HP1_INFO': 'g'}, inplace=True)

# merge dataframes on year, outerwise
df = pd.merge(inflation, interest_rates, on='year', how='outer')
# df = pd.merge(df, gdp, on='year', how='outer') --> unsmoothed GDP
df = pd.merge(df, gdp_smooth, on='year', how='outer') # --> smoothed GDP
df = pd.merge(df, deficit, on='year', how='outer')
df = pd.merge(df, interest_outlays, on='year', how='outer')
df['s'] = df['Federal Deficit'] + df['Interest Outlays']
df['r'] = df['r'] - df['Inflation (10-Year Expected)']

# restrict to variables of interest 
df = df[['year', 'g', 'r', 's']].dropna()
# convert every var into decimal
for var in ['g', 'r', 's']:
    df[var] = df[var] / 100
# now plot our variables of interest over time from 1992 to present 
df = df[df['year'] >= 1992]
plt.figure(figsize=(12, 8))
plt.plot(df['year'], df['g'], label='g (GDP Growth)', color='blue')
plt.plot(df['year'], df['r'], label='r (Interest Rate)', color='red')
plt.plot(df['year'], df['s'], label='s (Primary Deficit)', color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
# add vertical line at 2000, which is start of our subsample to calculate mean 
plt.axvline(x=2000, color='gray', linestyle='--', linewidth=0.8, label='Boundary Point')
plt.xticks(rotation=45)
plt.title('Historical Trends of r, g, and s (1992-Present)')
plt.legend(loc='best', fontsize='x-large')
plt.grid()
plt.tight_layout()
plt.savefig(output / 'historical_trends_rg_s.png', dpi=300)
plt.show()

# model estimation and calibration
df = df[df['year'] >= 2000]  # use data from 2000 onwards for calibration
df['rg'] = df['r'] - df['g']
au_dict = {
    '1': np.mean(df['rg']),
    '2': -1 * np.mean(df['rg']),
    '3': 0,
}

# now, we want to bring in an exogenous vector of values for a_s
forecasts = pd.read_csv(clean_data / 'master_projections_cleaned.csv')
forecasts['year'] = pd.to_datetime(forecasts['date']).dt.year
forecasts = forecasts.groupby('year')['s (tbl senate, permanent)'].mean().reset_index()
# restrict to the next 10 years 
forecasts = forecasts[forecasts['year'] >= 2025]
forecasts = forecasts[forecasts['year'] <= 2035]
# convert to vector
a_s = forecasts['s (tbl senate, permanent)'].values
a_s = a_s / 100  # convert to decimal

# exogenously define s_x to be 1\%
s_x = 0.01
s_s = 0.01
x0 = 0
n_years = 10
b0 = 1.21

for c_val, label in zip(
    [0.33, 0.67],
    ['irresponsible', 'responsible']
):
    c_dict = {}
    for scenario, a_u in au_dict.items():
        print(f"{scenario}: Estimated a_u (mean r - g): {a_u:.4f}")
        # find variable equal to difference between rg and a_u
        df['e_u'] = df['rg'] - a_u
        s_u = np.std(df['e_u'])
        print(f"Estimated s_u (std dev of e_u): {s_u:.4f}")
        # run simulations with these parameters
        '''
        now i want to plot a few things:
        1. 25th percentile, median, and 75th percentile of b for each year
        2. mean slope of b
        3. mean curvature of b

        then, separately, i want to plot the distribution of curvatures of b as a histogram
        '''
        results = []
        for sim in range(n_simulations):
            x = np.zeros(n_years)
            u = np.zeros(n_years)
            s = np.zeros(n_years)
            b = np.zeros(n_years)
            rg = np.zeros(n_years)
            x[0] = x0
            b[0] = b0
            u[0] = a_u + np.random.normal(0, s_u)
            rg[0] = x[0] + u[0]
            s[0] = (1 - c_val) * a_s[0] + c_val * (rg[0] * b[0]) + np.random.normal(0, s_s)

            for t in range(1, n_years):
                x[t] = x[t-1] + np.random.normal(0, s_x)
                u[t] = a_u + np.random.normal(0, s_u)
                rg[t] = x[t] + u[t]
                s[t] = (1 - c_val) * a_s[t] + c_val * (rg[t] * b[t-1]) + np.random.normal(0, s_s)
                b[t] = b[t-1] + rg[t] * b[t-1] - s[t]

            for t in range(n_years):
                results.append({
                    "year": t+1,
                    "rg": rg[t],
                    "b": b[t],
                    "sim": sim,          # 👈 add this

                })
        temp = pd.DataFrame(results)
        c_dict[scenario] = temp

    # plot median and percentiles of rg 
    plt.figure(figsize=(12, 8))
    for scenario, color in zip(au_dict.keys(), ['orange', 'green', 'blue']):
        subset = c_dict[scenario]
        subset = subset.groupby(['year']).agg(
            rg_25th = ('rg', lambda x: np.percentile(x, 25)),
            rg_median = ('rg', 'median'),
            rg_75th = ('rg', lambda x: np.percentile(x, 75)),
        ).reset_index()
        plt.plot(subset['year'], subset['rg_median'], label=f'a_u = {au_dict[scenario]:.2f} Median', color=color)
        plt.fill_between(subset['year'], subset['rg_25th'], subset['rg_75th'], alpha=0.2, 
                         color=color)
    plt.xlabel('Year')
    plt.ylabel('(r-g)')
    plt.legend(loc='best', fontsize='x-large')
    plt.title(f'Median and Percentiles of (r-g) - {label.title()} Fiscal Regime')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # plot the median and percentiles of b
    plt.figure(figsize=(12, 8))
    scenarios = list(au_dict.keys())
    colors = ['orange', 'green', 'blue']
    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        subset = c_dict[scenario]
        # calculate slope and curvature
        subset['slope'] = subset.groupby('sim')['b'].transform(np.gradient)
        subset['curvature'] = subset.groupby('sim')['slope'].transform(np.gradient)
        c_dict[scenario] = subset
        subset = subset.groupby(['year']).agg(
            b_25th=('b', lambda x: np.percentile(x, 25)),
            b_median=('b', 'median'),
            b_75th=('b', lambda x: np.percentile(x, 75)),
        ).reset_index()
        plt.plot(subset['year'], subset['b_median'], label=f'a_u = {au_dict[scenario]:.2f} Median', color=color)
        plt.fill_between(subset['year'], subset['b_25th'], subset['b_75th'], alpha=0.2, color=color)
        # while we're here, calculate slope and curvature
    plt.ylim(b0-0.01, 1.60)
    plt.axhline(y=b0, color='black', linestyle='--', label=f'Initial Debt Level ({b0})')
    plt.xlabel('Year')
    plt.ylabel('Debt Level')
    plt.legend(loc='best', fontsize='x-large')
    plt.title(f'Median and Percentiles of Debt Level - {label.title()} Fiscal Regime')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{graphics_path}/sdsa_debt_{label}.png', dpi=300)
    plt.show()

    # now calculate the median/25th/75th percentile slope/curvature
    for var in ['slope', 'curvature']:
        plt.figure(figsize=(12, 8))
        for scenario, color in zip(scenarios, colors):
            subset = c_dict[scenario]
            subset = subset.groupby('year').agg(
                _25th=(var, lambda x: np.percentile(x, 25)),
                median=(var, 'median'),
                _75th=(var, lambda x: np.percentile(x, 75)),
            ).reset_index()
            plt.plot(subset['year'], subset['median'], 
                     label=f'a_u = {au_dict[scenario]:.2f} {var.title()}', 
                     color=color)
            plt.fill_between(subset['year'], subset['_25th'], subset['_75th'], alpha=0.2, color=color)
        plt.axhline(y=0, color='black', linestyle='--', label='Zero Line')
        if var == 'slope':
            plt.ylim(-0.02, 0.06)
        else:
            plt.ylim(-0.01, 0.01)
        plt.xlabel('Year')
        plt.ylabel(var.title())
        plt.legend(loc='best', fontsize='x-large')
        plt.title(f'Mean {var.title()} of Debt Level - {label.title()} Fiscal Regime')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{graphics_path}/sdsa_debt_{label}_{var}.png', dpi=300)
        plt.show()
    
    # now plot distribution of curvature 
    plt.figure(figsize=(12, 8))
    for scenario, color in zip(scenarios, colors):
        subset = c_dict[scenario]
        plt.hist(subset['curvature'], bins=30, alpha=0.5, 
                 label=f'a_u = {au_dict[scenario]:.2f}', color=color)
    plt.xlabel('Curvature of Debt Level')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Curvature of Debt Level - {label.title()} Fiscal Regime')
    plt.legend(loc='best', fontsize='x-large')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{graphics_path}/sdsa_debt_{label}_curvature_distribution.png', dpi=300)
    plt.show()

################################################################################
## 4. Enrichment 1 - separate modeling for r and g 
################################################################################
graphics_path = output / 'sdsa_enrichment_1'
def simulate_scenario(c_val, a_s_vec, a_ug, r_star, beta_r, rho, sigma, s_g, s_x, s_r, s_s,
                      x0=0, r0=None, b0=1.21, n_years=10, n_simulations=5000, a_u=None, label=""):

    results = []

    for sim in range(n_simulations):
        # Initialize arrays
        x = np.zeros(n_years)
        e_g = np.zeros(n_years)
        e_x = np.zeros(n_years)
        e_r = np.zeros(n_years)
        e_s = np.zeros(n_years)
        g = np.zeros(n_years)
        r = np.zeros(n_years)
        r_av = np.zeros(n_years)
        s = np.zeros(n_years)
        b = np.zeros(n_years)

        # Initial values
        x[0] = x0
        g[0] = a_ug[0] + x[0] + np.random.normal(0, s_g)
        r[0] = r0
        r_av[0] = r[0]
        b[0] = b0
        s[0] = (1 - c_val) * a_s_vec[0] + c_val * (r_av[0] - g[0]) * b[0] + np.random.normal(0, s_s)

        for t in range(1, n_years):
            # Draw shocks
            e_g[t] = np.random.normal(0, s_g)
            e_x[t] = np.random.normal(0, s_x)
            e_r[t] = np.random.normal(0, s_r)
            e_s[t] = np.random.normal(0, s_s)

            # Update x and g
            x[t] = x[t - 1] + e_x[t]
            g[t] = a_ug[t] + x[t] + e_g[t]

            # Update r with AR(1) + feedback
            b_lag = b[t - 1]
            b_lag2 = b[t - 2] if t >= 2 else b[t - 1]
            r[t] = r_star + beta_r * b_lag + rho * (r[t - 1] - r_star - beta_r * b_lag2) + e_r[t]

            # Update smoothed r
            r_av[t] = sigma * r_av[t - 1] + (1 - sigma) * r[t]

            # Update s
            s[t] = (1 - c_val) * a_s_vec[t] + c_val * (r_av[t] - g[t]) * b[t - 1] + e_s[t]

            # Update debt
            b[t] = b[t - 1] + ((r_av[t] - g[t]) / (1 + g[t])) * b[t - 1] - s[t]

        for t in range(n_years):
            results.append({
                "year": t + 1,
                "sim": sim,
                "b": b[t],
                "r": r[t],
                "g": g[t],
                "r_av": r_av[t],
                "s": s[t],
                "c": c_val,
                "label": label
            })

    return pd.DataFrame(results)

# bring in CBO growth projections
cbo_forecasts = pd.read_csv(f'{clean_data}/master_projections_cleaned.csv')
# extract year and collapse growth average on year 
cbo_forecasts['year'] = pd.to_datetime(cbo_forecasts['date']).dt.year
cbo_forecasts = cbo_forecasts.groupby('year')['g (cbo baseline)'].mean().reset_index()
# restrict to the next 10 years
cbo_forecasts = cbo_forecasts[cbo_forecasts['year'] >= 2025]
cbo_forecasts = cbo_forecasts[cbo_forecasts['year'] <= 2035]
# convert to vector
a_ug = cbo_forecasts['g (cbo baseline)'].values
a_ug = a_ug / 100  # convert to decimal

# set parameters
r_star = 0.01
rho = 0.85
sigma = 0.8
s_g = 0.005
s_x = 0.002
s_r = 0.005
s_s = 0.01
r0 = 0.01  # initial interest rate
b0 = 1.21  # initial debt level
 
# for different levels of c, run the simulation 
d_dict = {
    'irresponsible': 0.33,
    'responsible': 0.67
}
beta_r_dict = {
    'No Feedback': 0.0,
    '2 bps': 0.02,
    '3 bps': 0.03,
    '5 bps': 0.05,
}

# Precompute global y-limits for each plot type
ylim_dict = {
    "r": [],
    "rg": [],
    "b": [],
    "slope": [],
    "curvature": []
}

# First pass to collect limits
for c_val in d_dict.values():
    for beta_r in beta_r_dict.values():
        df_sim = simulate_scenario(
            c_val=c_val,
            a_s_vec=a_s,
            a_ug=a_ug,
            r_star=r_star,
            beta_r=beta_r,
            rho=rho,
            sigma=sigma,
            s_g=s_g,
            s_x=s_x,
            s_r=s_r,
            s_s=s_s,
            x0=x0,
            r0=r0,
            b0=b0,
            n_years=n_years,
            n_simulations=500,  # use fewer sims for speed in limit detection
            a_u=None,
            label=""
        )
        df_sim["rg"] = df_sim["r"] - df_sim["g"]
        df_sim["slope"] = df_sim.groupby("sim")["b"].transform(np.gradient)
        df_sim["curvature"] = df_sim.groupby("sim")["slope"].transform(np.gradient)
        for var in ylim_dict:
            ylim_dict[var].extend(df_sim[var].values)

# Final global limits
ylim_bounds = {k: (np.percentile(v, 0.5), np.percentile(v, 99.5)) for k, v in ylim_dict.items()}

for label, c_val in d_dict.items():
    print(f"Running simulations for c = {c_val} ({label.title()})")
    sim_results = {}
    for beta_r_label, beta_r in beta_r_dict.items():
        print(f"Running simulation for c = {c_val}, beta_r = {beta_r } ({beta_r_label})")
        sim_results[beta_r] = simulate_scenario(
            c_val=c_val,
            a_s_vec=a_s,
            a_ug=a_ug,
            r_star=r_star,
            beta_r=beta_r,
            rho=rho,
            sigma=sigma,
            s_g=s_g,
            s_x=s_x,
            s_r=s_r,
            s_s=s_s,
            x0=x0,
            r0=r0,
            b0=b0,
            n_years=n_years,
            n_simulations=n_simulations,
            a_u=None,
            label=label
        )
    # now plot the results 
    # first, plot path of interest rates 
    plt.figure(figsize=(12, 8))
    for beta_r_label, beta_r in beta_r_dict.items():
        df_sim = sim_results[beta_r]
        df_mean = df_sim.groupby('year').agg(
            r_median=('r', 'median'),
            r_25th=('r', lambda x: np.percentile(x, 25)),
            r_75th=('r', lambda x: np.percentile(x, 75))
        ).reset_index()
        plt.plot(df_mean['year'], df_mean['r_median'], label=f'beta_r = {beta_r} ({beta_r_label})')
        plt.fill_between(df_mean['year'], df_mean['r_25th'], df_mean['r_75th'], alpha=0.2)
    plt.axhline(y=r_star, color='black', linestyle='--', label=f'Long-Term r* ({r_star})')
    plt.xlabel('Year')
    plt.ylabel('Interest Rate (r)')
    plt.title(f'Median and Percentiles of Interest Rate (r) - {label.title()} Fiscal Regime')
    plt.legend(loc='best', fontsize='x-large')
    plt.grid()
    plt.ylim(*ylim_bounds["r"])
    plt.tight_layout()
    plt.savefig(graphics_path / f'sdsa_interest_rate_{label}.png', dpi=300)
    plt.show()

    # next plot path of r-g 
    plt.figure(figsize=(12, 8))
    for beta_r_label, beta_r in beta_r_dict.items():
        df_sim = sim_results[beta_r]
        df_sim['rg'] = df_sim['r'] - df_sim['g']
        df_mean = df_sim.groupby('year').agg(
            rg_median=('rg', 'median'),
            rg_25th=('rg', lambda x: np.percentile(x, 25)),
            rg_75th=('rg', lambda x: np.percentile(x, 75))
        ).reset_index()
        plt.plot(df_mean['year'], df_mean['rg_median'], label=f'beta_r = {beta_r} ({beta_r_label})')
        plt.fill_between(df_mean['year'], df_mean['rg_25th'], df_mean['rg_75th'], alpha=0.2)
    plt.axhline(y=0, color='black', linestyle='--', label='Zero Line')
    plt.xlabel('Year')
    plt.ylabel('Interest-Growth Differential (r - g)')
    plt.title(f'Median and Percentiles of Interest-Growth Differential (r - g) - {label.title()} Fiscal Regime')
    plt.legend(loc='best', fontsize='x-large')
    plt.grid()
    plt.ylim(*ylim_bounds["rg"])
    plt.tight_layout()
    plt.savefig(graphics_path / f'sdsa_rg_{label}.png', dpi=300)
    plt.show()

    # now plot the path of debt 
    plt.figure(figsize=(12, 8))
    for beta_r_label, beta_r in beta_r_dict.items():
        df_sim = sim_results[beta_r]
        df_mean = df_sim.groupby('year').agg(
            b_median=('b', 'median'),
            b_25th=('b', lambda x: np.percentile(x, 25)),
            b_75th=('b', lambda x: np.percentile(x, 75))
        ).reset_index()
        plt.plot(df_mean['year'], df_mean['b_median'], label=f'beta_r = {beta_r} ({beta_r_label})')
        plt.fill_between(df_mean['year'], df_mean['b_25th'], df_mean['b_75th'], alpha=0.2)
    plt.axhline(y=b0, color='black', linestyle='--', label=f'Initial Debt Level ({b0})')
    plt.xlabel('Year')
    plt.ylabel('Debt Level (b)')
    plt.title(f'Median and Percentiles of Debt Level (b) - {label.title()} Fiscal Regime')
    plt.legend(loc='best', fontsize='x-large')
    plt.grid()
    plt.ylim(*ylim_bounds["b"])
    plt.tight_layout()
    plt.savefig(graphics_path / f'sdsa_debt_{label}.png', dpi=300)
    plt.show()

    # now plot the slope and curvature of debt
    plt.figure(figsize=(12, 8))
    for beta_r_label, beta_r in beta_r_dict.items():
        df_sim = sim_results[beta_r]
        df_sim['slope'] = df_sim.groupby('sim')['b'].transform(np.gradient)
        df_mean = df_sim.groupby('year').agg(
            slope_median=('slope', 'median'),
            slope_25th=('slope', lambda x: np.percentile(x, 25)),
            slope_75th=('slope', lambda x: np.percentile(x, 75)),\
        ).reset_index()
        plt.plot(df_mean['year'], df_mean['slope_median'], label=f'beta_r = {beta_r} ({beta_r_label})')
        plt.fill_between(df_mean['year'], df_mean['slope_25th'], df_mean['slope_75th'], alpha=0.2)
    plt.axhline(y=0, color='black', linestyle='--', label='_nolegend_')
    plt.xlabel('Year')
    plt.ylabel('Slope of Debt Level (b)')
    plt.title(f'Median and Percentiles of Slope of Debt Level (b) - {label.title()} Fiscal Regime')
    plt.legend(loc='best', fontsize='x-large')
    plt.grid()
    plt.ylim(*ylim_bounds["slope"])
    plt.tight_layout()
    plt.savefig(graphics_path / f'sdsa_debt_slope_{label}.png', dpi=300)
    plt.show()

    # now plot the slope and curvature of debt
    plt.figure(figsize=(12, 8))
    for beta_r_label, beta_r in beta_r_dict.items():
        df_sim = sim_results[beta_r]
        df_sim['curvature'] = df_sim.groupby('sim')['slope'].transform(np.gradient)
        df_mean = df_sim.groupby('year').agg(
            curvature_median=('curvature', 'median'),
            curvature_25th=('curvature', lambda x: np.percentile(x, 25)),
            curvature_75th=('curvature', lambda x: np.percentile(x, 75))
        ).reset_index()
        plt.plot(df_mean['year'], df_mean['curvature_median'], label=f'beta_r = {beta_r} ({beta_r_label})')
        plt.fill_between(df_mean['year'], df_mean['curvature_25th'], df_mean['curvature_75th'], alpha=0.2)
    plt.axhline(y=0, color='black', linestyle='--', label='_nolegend_')
    plt.xlabel('Year')
    plt.ylabel('Curvature of Debt Level (b)')
    plt.title(f'Median and Percentiles of Curvature of Debt Level (b) - {label.title()} Fiscal Regime')
    plt.legend(loc='best', fontsize='x-large')
    plt.grid()
    plt.ylim(*ylim_bounds["curvature"])
    plt.tight_layout()
    plt.savefig(graphics_path / f'sdsa_debt_curvature_{label}.png', dpi=300)
    plt.show()