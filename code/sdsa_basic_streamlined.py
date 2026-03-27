# ============================================
# ENRICHMENT 1 / FULLY-ENRICHED SDSA SIMULATOR
# with FIRST-DIFFERENCE r law of motion
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path
from fredapi import Fred
import statsmodels.api as sm
import seaborn as sns

# -------------------------------------------------
# Paths, plotting style, and helpers (unchanged)
# -------------------------------------------------
home = Path.home()
work_dir = (home / 'debt_sustainability_project')
data = (work_dir / 'data' / 'sdsa')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
output = (work_dir / 'output' / 'sdsa' / 'graphics')
output.mkdir(parents=True, exist_ok=True)

plt.style.use('mahoney_lab.mplstyle')

fred = Fred(api_key='8905b2f5faefd705486e644f09bb8088')

def get_fred_series(series_id, series_name):
    s = fred.get_series(series_id)
    df = pd.DataFrame(s, columns=[series_name])
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    return df

# smooth derivatives utility you already had
from scipy.signal import savgol_filter
def _sg_deriv(series, window=9, poly=2, deriv=1, use_log=False):
    x = series.to_numpy()
    if use_log:
        if (x <= 0).any():
            raise ValueError("use_log=True requires strictly positive series.")
        x = np.log(x)
    n = len(x)
    w = window if window % 2 == 1 else window - 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= poly:
        w = poly + 1 if (poly + 1) % 2 == 1 else poly + 2
        w = min(w, n if n % 2 == 1 else n - 1)
    return pd.Series(
        savgol_filter(x, w, poly, deriv=deriv, delta=1.0, mode='interp'),
        index=series.index
    )

# -------------------------------------------------
# Configuration: year ranges and initial conditions
# -------------------------------------------------
HIST_START  = 2000     # start of historical debt series
HIST_END    = 2025     # last year of observed data
PROJ_START  = 2026     # first year of CBO projection window
PROJ_END    = 2035     # last year of CBO projection window
SIM_START   = 2026     # calendar year the simulation begins

# -------------------------------------------------
# Inputs for Enrichment 1 (growth baseline and s path)
# -------------------------------------------------
# Load master projections once
master_proj = pd.read_csv(clean_data / 'master_projections_cleaned.csv')
master_proj['year'] = pd.to_datetime(master_proj['date']).dt.year
proj_mask = (master_proj['year'] >= PROJ_START) & (master_proj['year'] <= PROJ_END)

# CBO growth baseline -> a_ug (levels, decimal)
cbo_forecasts = (master_proj[proj_mask].groupby('year')['g (cbo baseline)']
                 .mean().reset_index())
a_ug = (cbo_forecasts['g (cbo baseline)'].values) / 100.0

# CBO interest rate baseline -> a_ur (levels, decimal)
cbo_rates = (master_proj[proj_mask].groupby('year')['r (cbo baseline)']
             .mean().reset_index())
a_ur = (cbo_rates['r (cbo baseline)'].values) / 100.0

# read in higher TFP growth scenario from CBO updated values
cbo_ai = pd.read_excel(clean_data / 'cbo_ai_projections.xlsx', sheet_name='higher_tfp_data')
cbo_ai = cbo_ai[(cbo_ai['year'] >= PROJ_START) &
                (cbo_ai['year'] <= PROJ_END)]
a_ug_ai = (cbo_ai['g (cbo ai)'].values) / 100.0

# Senate TBL baseline for s -> a_s (levels, decimal)
a_s_alternative = (master_proj[proj_mask].groupby('year')['s (cbo baseline)']
                   .mean().reset_index())
a_s_alternative = (a_s_alternative['s (cbo baseline)'].values) / 100.0

a_s_df = (master_proj[proj_mask].groupby('year')['s (tbl senate, permanent)']
          .mean().reset_index())
a_s = (a_s_df['s (tbl senate, permanent)'].values) / 100.0

debt_hist = get_fred_series('FYGFGDQ188S', 'debt_pct_gdp')
debt_hist = debt_hist[debt_hist['date'] >= f'{HIST_START}-01-01'].copy()
debt_hist['b_hist'] = debt_hist['debt_pct_gdp'] / 100.0
debt_hist = debt_hist[debt_hist['date'].dt.month == 10].copy()
debt_hist['calendar_year'] = debt_hist['date'].dt.year
debt_hist = debt_hist[(debt_hist['calendar_year'] >= HIST_START) &
                      (debt_hist['calendar_year'] <= HIST_END)][['calendar_year', 'b_hist']].dropna()

growth = get_fred_series('A191RL1Q225SBEA', 'gdp_growth_rate') # quarterly, percent (real)
interest = get_fred_series('REAINTRATREARAT10Y', 'interest_rate') # monthly, percent (real)
snowball_hist = growth.merge(interest, on='date', how='outer')
# convert to calendar year
snowball_hist = snowball_hist[snowball_hist['date'].dt.month == 10].copy()
snowball_hist['calendar_year'] = snowball_hist['date'].dt.year
snowball_hist = snowball_hist[(snowball_hist['calendar_year'] >= HIST_START) &
                                (snowball_hist['calendar_year'] <= HIST_END)].copy()
snowball_hist = debt_hist.merge(snowball_hist, on='calendar_year', how='left')
snowball_hist['snowball'] = (snowball_hist['interest_rate'] - snowball_hist['gdp_growth_rate']) / 100.0 * snowball_hist['b_hist']

# -------------------------------------------------
# Core simulator (CBO-baseline r with debt feedback)
# -------------------------------------------------
def simulate_scenario(
    c_val,
    a_s_vec, a_ug, a_ur,
    beta_r, rho, sigma,
    s_g, s_x, s_r, s_s,
    x0=0.0, r_av0=None, b0=0.9818154,
    n_years=10, n_simulations=20000, label=""
):
    """
    Fully-enriched SDSA:
      - g_t = a_ug[t] + x_t + e_g           (CBO growth baseline + random walk + shock)
      - x_t random walk
      - r_t = a_ur[t] + z_t                  (CBO rate baseline + deviation)
        where z_t evolves via first-difference debt feedback:
            Δz_t = β_r(Δb_{t-1} - ρΔb_{t-2}) + ρΔz_{t-1} + η_t,  η_t = ε_t - ε_{t-1}
      - r_av,t = σ r_av,t-1 + (1-σ) r_t     (pass-through to avg rate on debt)
      - s_t = (1-c) a_s[t] + c (r_av,t - g_t) b_{t-1} + e_s
      - b_t = b_{t-1} + ((r_av,t - g_t)/(1+g_t)) b_{t-1} - s_t
    """
    results = []

    for sim in range(n_simulations):
        # state arrays
        x   = np.zeros(n_years)
        g   = np.zeros(n_years)
        z   = np.zeros(n_years)   # deviation of r from CBO baseline
        r   = np.zeros(n_years)
        r_av= np.zeros(n_years)
        s   = np.zeros(n_years)
        b   = np.zeros(n_years)

        # shocks (draw on the fly; we only need last ε for η_t)
        eps_r_prev = 0.0

        # initials: year 0 = first forecast year (2026), b0 = 2025 observed debt
        x[0]   = x0
        g[0]   = a_ug[0] + x[0] + np.random.normal(0, s_g)
        z[0]   = 0.0              # no deviation from CBO baseline at start
        r[0]   = a_ur[0] + z[0]
        r_av[0]= r_av0 if r_av0 is not None else r[0]
        s[0]   = (1 - c_val) * a_s_vec[0] + c_val * (r_av[0] - g[0]) * b0 + np.random.normal(0, s_s)
        b[0]   = b0 + ((r_av[0] - g[0]) / (1.0 + g[0])) * b0 - s[0]

        for t in range(1, n_years):
            # shocks
            e_g = np.random.normal(0, s_g)
            e_x = np.random.normal(0, s_x)
            eps_r = np.random.normal(0, s_r)
            e_s = np.random.normal(0, s_s)

            # x, g
            x[t] = x[t-1] + e_x
            g[t] = a_ug[t] + x[t] + e_g

            # debt feedback on r deviation (z)
            db_t_1 = (b[t-1] - b[t-2]) if t >= 2 else 0.0
            db_t_2 = (b[t-2] - b[t-3]) if t >= 3 else 0.0
            dz_t_1 = (z[t-1] - z[t-2]) if t >= 2 else 0.0
            eta_t  = eps_r - eps_r_prev

            # first-difference law on deviations from CBO baseline
            dz_t = beta_r * (db_t_1 - rho * db_t_2) + rho * dz_t_1 + eta_t
            z[t] = z[t-1] + dz_t
            r[t] = a_ur[t] + z[t]

            # pass-through to average rate paid on debt
            r_av[t] = sigma * r_av[t-1] + (1.0 - sigma) * r[t]

            # primary balance with fiscal feedback
            s[t] = (1 - c_val) * a_s_vec[t] + c_val * (r_av[t] - g[t]) * b[t-1] + e_s

            # debt accumulation (discrete approximation)
            b[t] = b[t-1] + ((r_av[t] - g[t]) / (1.0 + g[t])) * b[t-1] - s[t]

            # update previous epsilon for next eta_t
            eps_r_prev = eps_r

        # collect
        for t in range(n_years):
            results.append({
                "year": t + 1,
                "sim": sim,
                "b": b[t],
                "r": r[t],
                "r_av": r_av[t],
                "g": g[t],
                "s": s[t],
                "c": c_val,
                "label": label
            })

    return pd.DataFrame(results)

# -------------------------------------------------
# Parameters and run set
# -------------------------------------------------
np.random.seed(42)

# horizons and baseline levels
n_years = len(a_ug)
n_sims  = 5000
b0      = 0.9818154
r_av0   = 0.0125    # initial r_av: 2025 effective rate on existing debt

# stochastic vols
s_g = 0.005
s_x = 0.002
s_r = 0.005
s_s = 0.010

# smoothing / persistence
rho   = 0.158   # AR coefficient in Δr law (on Δr_{t-1})
sigma = 0.80   # pass-through to r_av

# fiscal reaction regimes (only enrichment runs)
d_dict = {
    'irresponsible': 0.00,
    'responsible'  : 0.15,
    'very responsible': 0.30
}

# β_r in pp per 1-pp debt (3 bps baseline)
beta_r_dict = {'3 bps': 0.0171}

# -------------------------------------------------
# Simulate, summarize, and plot (Enrichment 1 only)
# -------------------------------------------------
    # helper: compute median & IQR by year
def _band_by_year(df, var, time_col="calendar_year"):
    # fall back gracefully if calendar_year doesn't exist
    if time_col not in df.columns:
        if "year" in df.columns:
            time_col = "year"
        else:
            raise KeyError(f"Neither '{time_col}' nor 'year' found in df columns: {list(df.columns)}")

    g = (df.groupby(time_col)[var]
         .agg(median='median',
              p25=lambda x: np.percentile(x, 25),
              p75=lambda x: np.percentile(x, 75))
         .reset_index())

    # standardize output column name
    g = g.rename(columns={time_col: "time"})
    return g


def summarize_and_plot(sim_results_by_regime, graphics_path: Path, d_dict: dict,
                       b0: float, r_av0: float,
                       debt_hist: pd.DataFrame,
                       snowball_hist: pd.DataFrame,
                       sim_start_year: int = SIM_START):

    graphics_path.mkdir(parents=True, exist_ok=True)

    # 2025 anchor values from historical data (for connecting lines)
    anchor_year = sim_start_year - 1
    hist_2025 = snowball_hist[snowball_hist['calendar_year'] == anchor_year]
    anchor_rg = (hist_2025['interest_rate'].values[0] - hist_2025['gdp_growth_rate'].values[0]) / 100.0 if len(hist_2025) > 0 else 0.0
    anchor_snowball = hist_2025['snowball'].values[0] if len(hist_2025) > 0 else 0.0
    anchor_g = hist_2025['gdp_growth_rate'].values[0] / 100.0 if len(hist_2025) > 0 else 0.0

    ylim_store = {k: [] for k in ['g','r_av','rg','b','slope','curvature','interest_share','snowball']}
    enriched = {}

    for label, df_sim in sim_results_by_regime.items():
        df_sim = df_sim.copy()
        df_sim["calendar_year"] = sim_start_year + df_sim["year"] - 1
        df_sim['rg'] = df_sim['r'] - df_sim['g']
        df_sim['slope'] = df_sim.groupby('sim', group_keys=False)['b'].apply(
            lambda s: _sg_deriv(s, window=5, poly=2, deriv=1))
        df_sim['curvature'] = df_sim.groupby('sim', group_keys=False)['b'].apply(
            lambda s: _sg_deriv(s, window=5, poly=2, deriv=2))
        df_sim['interest_share'] = df_sim['r_av'] * df_sim.groupby('sim')['b'].shift(1).fillna(b0)
        df_sim['snowball'] = (df_sim['r_av'] - df_sim['g']) * df_sim.groupby('sim')['b'].shift(1).fillna(b0)

        # Prepend a 2025 anchor row for each sim so projected lines connect to history
        sims = df_sim['sim'].unique()
        anchor_rows = pd.DataFrame({
            'year': 0,
            'sim': sims,
            'b': b0,
            'r': anchor_rg + anchor_g,  # approximate r from historical r-g + g
            'r_av': r_av0,
            'g': anchor_g,
            's': 0.0,
            'c': df_sim['c'].iloc[0],
            'label': df_sim['label'].iloc[0],
            'calendar_year': anchor_year,
            'rg': anchor_rg,
            'slope': np.nan,
            'curvature': np.nan,
            'interest_share': r_av0 * b0,
            'snowball': anchor_snowball,
        })
        df_sim = pd.concat([anchor_rows, df_sim], ignore_index=True)

        for var in ylim_store:
            ylim_store[var].extend(df_sim[var].dropna().values)

        enriched[label] = df_sim

    ylim_store["b"].extend(debt_hist["b_hist"].values)
    ylim_store["snowball"].extend(snowball_hist["snowball"].dropna().values)

    # global axis ranges (robust to tails)
    ylim_bounds = {k: (np.percentile(v, 0.5), np.percentile(v, 99.5)) for k, v in ylim_store.items()}

    # overlay plots for each metric
    def _plot_overlay(var, title, ylab, yline=None, ylim_key=None, fname=None):
        plt.figure(figsize=(11,7))

        # Add historical debt series ONLY on the debt chart
        if var == "b":
            plt.plot(
                debt_hist["calendar_year"], debt_hist["b_hist"],
                color="black", linewidth=3.0, label="Historical (FRED, 2000–2025)"
            )
            plt.axvline(sim_start_year - 1, color="black", linestyle="--", linewidth=0.9, alpha=0.8,
                    label="_nolegend_")
        if var == "snowball":
            plt.plot(
                snowball_hist["calendar_year"], snowball_hist["snowball"],
                color="black", linewidth=3.0, label="Historical (2000–2025)"
            )
            plt.axvline(sim_start_year - 1, color="black", linestyle="--", linewidth=0.9, alpha=0.8,
                        label="_nolegend_")
        for label, c_val in d_dict.items():
            df_sim = enriched[label]
            var_name = var if var != "r" else "r_av"  # your existing convention
            g = _band_by_year(df_sim, var_name, time_col="calendar_year")

            plt.plot(g["time"], g["median"], label=f"{label.title()} (c={c_val:.2f})")
            plt.fill_between(g["time"], g["p25"], g["p75"], alpha=0.20)

        if yline is not None:
            plt.axhline(y=yline, color='black', linestyle='--', linewidth=0.9)

        plt.xlabel(''); plt.ylabel(ylab)
        plt.grid(True)

        if ylim_key:
            lo, hi = ylim_bounds[ylim_key]

        plt.legend(loc='best', fontsize='x-large')
        if fname:
            plt.tight_layout()
            plt.savefig(graphics_path / fname, dpi=300)
        plt.show()

    # now draw all overlays, WITH both regimes in each chart
    _plot_overlay('g',
                  'Growth (g): Median & IQR — Regimes',
                  'g',
                  ylim_key='g',
                  fname='sdsa_enrichment1_g_overlay.pdf')
    
    plt.figure(figsize=(11,7))
    for label, c_val in d_dict.items():
        if c_val != 0.15:
            continue
        df_sim = enriched[label]
        # Flags to ensure each category is labeled only once
        high_labeled = False
        low_labeled = False
        other_labeled = False
        for sim_id, df_sim_i in df_sim.groupby('sim'):
            mean_g = df_sim_i['g'].mean()
            if mean_g > 0.028:
                plt.plot(
                    df_sim_i['calendar_year'],
                    df_sim_i['g'],
                    color='green', alpha=0.2,
                    label='simulations w/ > 2.8% avg growth' if not high_labeled else ""
                )
                high_labeled = True
            elif mean_g < 0.0075:
                plt.plot(
                    df_sim_i['calendar_year'],
                    df_sim_i['g'],
                    color='red', alpha=0.2,
                    label='simulations w/ < 0.75% avg growth' if not low_labeled else ""
                )
                low_labeled = True
            else:
                plt.plot(
                    df_sim_i['calendar_year'],
                    df_sim_i['g'],
                    color='gray', alpha=0.01,
                    label='all other simulations' if not other_labeled else ""
                )
                other_labeled = True
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.9)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('real growth rate (g)')
    plt.grid(True)
    plt.legend(loc='best', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(graphics_path / 'sdsa_enrichment1_g_rates_responsible_only.pdf', dpi=300)
    plt.close()

    _plot_overlay('rg',
                  'Interest-Growth Differential (r - g): Median & IQR — Regimes',
                  'r - g',
                  yline=0.0,
                  ylim_key='rg',
                  fname='sdsa_enrichment1_rg_overlay.pdf')

    _plot_overlay('b',
                  'Debt (b): Median & IQR — Regimes',
                  'b',
                  yline=b0,
                  ylim_key='b',
                  fname='sdsa_enrichment1_b_overlay.pdf')

    _plot_overlay(
        'snowball',
        'Snowball term: (r - g) × b',
        'Snowball term',
        yline=0.0,
        ylim_key='snowball',
        fname='sdsa_enrichment1_snowball_overlay.pdf'
    )

    return enriched  # returns enriched frames in case you want to export

# ---- call it (note we pass d_dict and b0 so legends/lines show correctly)
sim_results_by_regime = {}
for label, c_val in d_dict.items():
    for beta_label, beta_r in beta_r_dict.items():
        df_sim = simulate_scenario(
            c_val=c_val, a_s_vec=a_s, a_ug=a_ug, a_ur=a_ur,
            beta_r=beta_r, rho=rho, sigma=sigma,
            s_g=s_g, s_x=s_x, s_r=s_r, s_s=s_s,
            x0=0.0, r_av0=r_av0, b0=b0,
            n_years=n_years, n_simulations=n_sims,
            label=f"{label} (β_r={beta_label})"
        )
        sim_results_by_regime[label] = df_sim  # one β shown here

# overlay plots with both regimes + c in legend text
enriched_frames = summarize_and_plot(sim_results_by_regime, output, d_dict, b0, r_av0,
                                     debt_hist=debt_hist, snowball_hist=snowball_hist,
                                     sim_start_year=SIM_START)
# (optional) export
all_sim_results = pd.concat(enriched_frames.values(), ignore_index=True)
all_sim_results.to_csv(output / 'sdsa_enrichment1_sim_results.csv', index=False)

# -------------------------------------------------
# Addendum - AI Boom Scenarios (0.5pp and 1pp Productivity Boost)
# -------------------------------------------------

# Each scenario provides AI GDP paths, growth rates, and changes in primary deficit (% of GDP)
# Baseline years 2025–2035
years = np.arange(PROJ_START, PROJ_END + 1)

# --- 0.5 pp productivity boost ---
a_ug_ai_05 = np.array([
    1.938, 1.988, 2.088, 2.250,
    2.411, 2.429, 2.466, 2.466, 2.451, 2.428
]) / 100
change_primary_deficit_pct_gdp_05 = np.array([
    -0.019924279, -0.064984957, -0.137927756, -0.246002998,
    -0.389646498, -0.545737236, -0.710450817, -0.885442132,
    -1.068350832, -1.262290796
]) / 100  # convert percentage points to decimals

# --- 1 pp productivity boost ---
a_ug_ai_10 = np.array([
    2.932, 2.906, 2.923, 2.997,
    3.064, 3.093, 3.141, 3.150, 3.145, 3.131
]) / 100
change_primary_deficit_pct_gdp_10 = np.array([
    -0.182794733, -0.445974749, -0.719954032,
    -1.019645652, -1.331048263, -1.657538639,
    -1.998705119, -2.348575081, -2.715048152,
    -3.097080547
]) / 100

# Baseline paths (CBO)
a_ug_baseline = a_ug
a_s_baseline = a_s

# Construct AI scenarios: shift the baseline primary balance by the improvement in deficit
a_s_ai_05 = a_s_baseline - change_primary_deficit_pct_gdp_05
a_s_ai_10 = a_s_baseline - change_primary_deficit_pct_gdp_10

# Bundle both scenarios
ai_scenarios = {
    "AI Boom (0.5pp Productivity)": {
        "a_ug": a_ug_ai_05,
        "a_s": a_s_ai_05
    },
    "AI Boom (1.0pp Productivity)": {
        "a_ug": a_ug_ai_10,
        "a_s": a_s_ai_10
    },
    "CBO Baseline": {
        "a_ug": a_ug_baseline,
        "a_s": a_s_baseline
    }
}

# Define fiscal responsibility cases
c_scenarios = {
    "Irresponsible (c=0.00)": 0.00,
    "Responsible (c=0.15)": 0.15,
    "Very Responsible (c=0.30)": 0.30
}

# Run simulations for all combinations
sim_results_by_ai = {}
for ai_label, params in ai_scenarios.items():
    for c_label, c_val in c_scenarios.items():
        df_sim = simulate_scenario(
            c_val=c_val, a_s_vec=params["a_s"], a_ug=params["a_ug"], a_ur=a_ur,
            beta_r=beta_r_dict["3 bps"], rho=rho, sigma=sigma,
            s_g=s_g, s_x=s_x, s_r=s_r, s_s=s_s,
            x0=0.0, r_av0=r_av0, b0=b0,
            n_years=n_years, n_simulations=n_sims,
            label=f"{ai_label} - {c_label}"
        )
        sim_results_by_ai[f"{ai_label} - {c_label}"] = df_sim

# ---- Precompute anchor values for connecting projections to history ----
hist_2025_ai = snowball_hist[snowball_hist['calendar_year'] == SIM_START - 1]
anchor_rg = (hist_2025_ai['interest_rate'].values[0] - hist_2025_ai['gdp_growth_rate'].values[0]) / 100.0 if len(hist_2025_ai) > 0 else 0.0
anchor_snowball = hist_2025_ai['snowball'].values[0] if len(hist_2025_ai) > 0 else 0.0
anchor_g = hist_2025_ai['gdp_growth_rate'].values[0] / 100.0 if len(hist_2025_ai) > 0 else 0.0

def _add_anchor_and_calendar(df_sim, sim_start=SIM_START, b0_val=b0,
                              r_av0_val=r_av0, anchor_snowball_val=anchor_snowball,
                              anchor_rg_val=anchor_rg, anchor_g_val=anchor_g):
    """Add calendar_year mapping and prepend 2025 anchor row to each sim."""
    df = df_sim.copy()
    df["calendar_year"] = sim_start + df["year"] - 1
    df['snowball'] = (df['r_av'] - df['g']) * df.groupby('sim')['b'].shift(1).fillna(b0_val)
    sims = df['sim'].unique()
    anchor = pd.DataFrame({
        'year': 0, 'sim': sims,
        'b': b0_val, 'r_av': r_av0_val, 'g': anchor_g_val,
        'r': anchor_rg_val + anchor_g_val,
        's': 0.0, 'c': df['c'].iloc[0], 'label': df['label'].iloc[0],
        'calendar_year': sim_start - 1,
        'snowball': anchor_snowball_val,
    })
    return pd.concat([anchor, df], ignore_index=True)

# ---- Plot overlays: Debt paths ----
for c_label, c_val in c_scenarios.items():
    filtered = {k: v for k, v in sim_results_by_ai.items() if c_label in k}

    # Debt overlay
    plt.figure(figsize=(11,7))
    plt.plot(debt_hist["calendar_year"], debt_hist["b_hist"],
             color="black", linewidth=3.0, label="Historical")
    for label, df_sim in filtered.items():
        df_a = _add_anchor_and_calendar(df_sim)
        g = _band_by_year(df_a, "b", time_col="calendar_year")
        plt.plot(g["time"], g["median"], label=label)
        plt.fill_between(g["time"], g["p25"], g["p75"], alpha=0.20)
    plt.axhline(y=b0, color="black", linestyle="--", linewidth=0.9)
    plt.ylim(b0 - 0.1, 1.7)
    plt.title("")
    plt.xlabel(""); plt.ylabel("Debt-to-GDP Ratio (b)")
    plt.grid(True)
    plt.legend(loc="best", fontsize="x-large")
    plt.tight_layout()
    plt.savefig(output / f"sdsa_enrichment1_b_overlay_ai_booms_{c_val:.2f}.pdf", dpi=300)
    plt.close()

    # Snowball overlay
    plt.figure(figsize=(11,7))
    sb = snowball_hist.dropna(subset=["snowball"])
    plt.plot(sb["calendar_year"], sb["snowball"],
             color="black", linewidth=3.0, label="Historical")
    for label, df_sim in filtered.items():
        df_a = _add_anchor_and_calendar(df_sim)
        g = _band_by_year(df_a, "snowball", time_col="calendar_year")
        plt.plot(g["time"], g["median"], label=label)
        plt.fill_between(g["time"], g["p25"], g["p75"], alpha=0.20)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.9)
    plt.title("")
    plt.xlabel(""); plt.ylabel("Snowball Term")
    plt.grid(True)
    plt.legend(loc="best", fontsize="x-large")
    plt.tight_layout()
    plt.savefig(output / f"sdsa_enrichment1_snowball_overlay_ai_booms_{c_val:.2f}.pdf", dpi=300)
    plt.close()

# ---- Optional overlay for growth ----
plt.figure(figsize=(11,7))
for ai_label, params in ai_scenarios.items():
    plt.plot(years, params["a_ug"], label=ai_label)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.9)
plt.title("")
plt.xlabel(""); plt.ylabel("a_ug (growth rate)")
plt.grid(True)
plt.legend(fontsize="x-large")
plt.tight_layout()
plt.savefig(output / "sdsa_enrichment1_a_ug_paths_ai_booms.pdf", dpi=300)
plt.close()

# ============================================================
# Distribution of 10-Year Debt Changes — AI vs Baseline (c = 0.15 only)
# ============================================================
# ── historical benchmark (FRED series) ───────────────────────
debt = get_fred_series('FYGFGDQ188S', 'Federal Debt Held by the Public')
debt.columns = debt.columns.str.lower()
debt.rename(columns={'federal debt held by the public': 'debt'}, inplace=True)
debt['debt'] = debt['debt'] / 100.0  # percent → decimal
debt_q4 = debt[debt['date'].dt.month == 10].sort_values('date').copy()
debt_q4['debt_10yr_change'] = debt_q4['debt'].diff(10) * 100  # 40 quarters = 10 years
historical_changes = debt_q4['debt_10yr_change'].dropna()

# ── helper to compute simulated 10y changes ──────────────────
def compute_ten_year_changes(df: pd.DataFrame) -> np.ndarray:
    """Compute 10-year change in debt/GDP (pp) per simulation."""
    changes = []
    for sim_id, df_sim in df.groupby("sim"):
        df_sim = df_sim.sort_values("year")
        change = (df_sim["b"].iloc[-1] - df_sim["b"].iloc[0]) * 100
        changes.append(change)
    return np.array(changes)

# ── collect three scenarios (c = 0.15 only) ──────────────────
target_labels = [
    "CBO Baseline - Irresponsible (c=0.00)",
    "CBO Baseline - Responsible (c=0.15)",
    "CBO Baseline - Very Responsible (c=0.30)",
    "AI Boom (0.5pp Productivity) - Irresponsible (c=0.00)",
]
distros = {}
for label in target_labels:
    if label not in sim_results_by_ai:
        print(f"Warning: {label} not found in results")
        continue
    distros[label] = compute_ten_year_changes(sim_results_by_ai[label])

# ============================================================
# Distribution of net interest rate payments — AI vs Baseline (c = 0.15 only)
# ============================================================
# 1. Pull historical net interest payments (% of GDP)
net_interest = get_fred_series('A091RC1Q027SBEA', 'net_interest')  
gdp = get_fred_series('GDP', 'gdp')
net_interest = net_interest.merge(gdp, on='date', how='inner')
net_interest['net_interest_pct_gdp'] = (net_interest['net_interest'] / net_interest['gdp']) * 100
net_interest = net_interest[['date', 'net_interest_pct_gdp', 'gdp']]
net_interest = net_interest[net_interest['date'].dt.month == 10]  # annualize using Q4 values
net_interest = net_interest.sort_values('date')
net_interest_df = net_interest.copy()
net_interest_series = net_interest['net_interest_pct_gdp'].dropna()

plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=["#1f77b4", "#9467bd", "#8c564b", "#7f7f7f", "#17becf", "#ff7f0e"]
)

# 2. Plot historical vs simulated distributions (transparent histograms)
plt.figure(figsize=(8,5))
graphing_labels = {
    "CBO Baseline - Irresponsible (c=0.00)": "Baseline",
    "CBO Baseline - Responsible (c=0.15)": "Responsive",
    "AI Boom (0.5pp Productivity) - Irresponsible (c=0.00)": "AI Boom",
    "CBO Baseline - Very Responsible (c=0.30)": "Very Responsive"
}
for ai_scenario in target_labels:
    if ai_scenario != "CBO Baseline - Very Responsible (c=0.30)":
        continue  # skip baseline here; plot last
    sim_df = sim_results_by_ai[ai_scenario]
    sim_df['net_interest_sim'] = (sim_df['r_av']) / (1 + sim_df['g']) * sim_df['b'].shift(1).fillna(b0)
    sim_df['net_interest_sim'] *= 100  # convert to percent
    sns.histplot(
        sim_df['net_interest_sim'],
        label=graphing_labels[ai_scenario],
        stat="density",
        element="step",
        fill=False,
        linewidth=1.5,
        alpha=1, linestyle='-'
    )
# one vertical line for current value 
current_value = net_interest_df['net_interest_pct_gdp'].iloc[-1]
plt.axvline(x=current_value, color='black', linestyle='--', linewidth=0.9, 
            label='_nolegend_')
# another vertical line for max historical value 
max_historic = net_interest_series.max()
plt.axvline(x=max_historic, color='gray', linestyle='--', linewidth=0.9, 
            label='_nolegend_')
# figure out standard deviation of historical series
std_historic = net_interest_series.std()
plt.axvline(x=max_historic + std_historic, color='blue', linestyle='--', linewidth=0.9, 
            label='_nolegend_')
# shade everything to the right of max historical value
plt.axvspan(max_historic + std_historic, plt.xlim()[1], color='red', alpha=0.1)
# print out share of simulations to the right of max historical value 
for ai_scenario in target_labels:
    if ai_scenario not in sim_results_by_ai:
        continue
    sim_df = sim_results_by_ai[ai_scenario]
    sim_df['net_interest_sim'] = (sim_df['r_av']) / (1 + sim_df['g']) * sim_df['b'].shift(1).fillna(b0)
    sim_df['net_interest_sim'] *= 100  # convert to percent
    pct_above = (sim_df['net_interest_sim'] > max_historic).mean() * 100
    print(f"{graphing_labels[ai_scenario]}: Percentage of simulations with net interest > {max_historic:.2f}%: {pct_above:.2f}%")
    # re-do w/ one std. above historic max
    pct_above_std = (sim_df['net_interest_sim'] > (max_historic + std_historic)).mean() * 100
    print(f"{graphing_labels[ai_scenario]}: Percentage of simulations with net interest > {max_historic + std_historic:.2f}%: {pct_above_std:.2f}%")
plt.title("")
plt.xlabel("Interest (% of GDP)")
plt.ylabel("Likelihood")
# ============================================================
# CUSTOM LEGEND WITH SPACER + CLEAN VERTICAL-LINE HANDLES
# ============================================================
# get histogram handles first
handles, labels = plt.gca().get_legend_handles_labels()
# spacer (draws nothing, creates vertical gap)
spacer = mpatches.Patch(color='none', label="")
# custom handles for vertical lines
h_current = mlines.Line2D([], [], color='black', linestyle='--', linewidth=0.9,
                          label="Level Today")
h_max = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=0.9,
                      label="Historical Max")
h_danger = mlines.Line2D([], [], color='blue', linestyle='--', linewidth=0.9,
                         label="Danger Zone")
# combine them: (1) all histograms, (2) spacer, (3) all vertical-line items
handles = handles + [spacer, h_current, h_max, h_danger]
labels = labels + ["", "Level Today", "Historical Max", "Danger Zone"]
plt.legend(handles, labels, fontsize='x-large', loc='best')
plt.tight_layout()
plt.show()

# ============================================================
# Effective interest rate comparison (historical vs simulated)
# ============================================================

debt_gdp = get_fred_series('FYGFGDQ188S', 'debt_held_by_public')  # in percent of GDP
growth = get_fred_series('A191RL1Q225SBEA', 'real_gdp_growth_rate')  # quarterly real GDP growth rate

net_interest_df = (
    net_interest_df
    .merge(debt_gdp, on='date', how='inner')
    .merge(growth, on='date', how='inner')
)
net_interest_df['effective_r'] = net_interest_df['net_interest_pct_gdp'] / net_interest_df['debt_held_by_public']

plt.figure(figsize=(8,5))
sns.histplot(
    net_interest_df['effective_r'],
    label="Historical",
    stat="density",
    element="step",
    fill=False,
    linewidth=2.5,
    alpha=1
)

for ai_scenario in target_labels:
    if ai_scenario not in sim_results_by_ai:
        continue
    sim_df = sim_results_by_ai[ai_scenario]
    sns.histplot(
        sim_df['r_av'],
        label=graphing_labels[ai_scenario],
        stat="density",
        element="step",
        fill=False,
        linewidth=1.5,
        alpha=1, linestyle='--'
    )
plt.ylim(0, 200)
plt.title("")
plt.xlabel("Effective Interest Rate (r_av)")
plt.ylabel("Density (Relative Frequency)")
plt.legend(loc='best', fontsize='x-large')
plt.tight_layout()
plt.close()

# now compare the debt accumulation snowball term (r^av - g) / (1 + g) * b[-1] to this same term applied historically
# effective_r is decimal; real_gdp_growth_rate and debt_held_by_public are in percent → convert
net_interest_df['debt_snowball_term'] = (
    (net_interest_df['effective_r'] - net_interest_df['real_gdp_growth_rate'] / 100) /
    (1 + net_interest_df['real_gdp_growth_rate'] / 100)
) * (net_interest_df['debt_held_by_public'] / 100)
plt.figure(figsize=(8,5))
sns.histplot(
    net_interest_df['debt_snowball_term'] * 100,
    label="Historical",
    stat="density",
    element="step",
    fill=False,
    linewidth=2.5,
    alpha=1
)
for ai_scenario in target_labels:
    if ai_scenario not in sim_results_by_ai:
        continue
    sim_df = sim_results_by_ai[ai_scenario]
    sim_df['debt_snowball_sim'] = (
        (sim_df['r_av'] - sim_df['g']) / (1 + sim_df['g'])
    ) * sim_df['b'].shift(1).fillna(b0) * 100  # convert to percent
    sns.histplot(
        sim_df['debt_snowball_sim'],
        label=graphing_labels[ai_scenario],
        stat="density",
        element="step",
        fill=False,
        linewidth=1.5,
        alpha=1, linestyle='--'
    )
plt.title("")
plt.xlabel("Debt Snowball Term")
plt.ylabel("Density (Relative Frequency)")
plt.legend(loc='best', fontsize='x-large')
plt.tight_layout()
plt.show()

# now compare r_av - g directly (both in decimal)
plt.figure(figsize=(8,5))
sns.histplot(
    net_interest_df['effective_r'] - net_interest_df['real_gdp_growth_rate'] / 100,
    label="Historical",
    stat="density",
    element="step",
    fill=False,
    linewidth=2.5,
    alpha=1
)
for ai_scenario in target_labels:
    if ai_scenario not in sim_results_by_ai:
        continue
    sim_df = sim_results_by_ai[ai_scenario]
    sim_df['rg_sim'] = (sim_df['r_av'] - sim_df['g']) * 100  # convert to percent
    sns.histplot(
        sim_df['rg_sim'],
        label=graphing_labels[ai_scenario],
        stat="density",
        element="step",
        fill=False,
        linewidth=1.5,
        alpha=1, linestyle='--'
    )
plt.title("")
plt.xlabel("r_av - g")
plt.ylabel("Density (Relative Frequency)")
plt.legend(loc='best', fontsize='x-large')
plt.tight_layout()
plt.close()