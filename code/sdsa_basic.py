# ============================================
# ENRICHMENT 1 / FULLY-ENRICHED SDSA SIMULATOR
# with FIRST-DIFFERENCE r law of motion
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# Inputs for Enrichment 1 (growth baseline and s path)
# -------------------------------------------------
# CBO growth baseline -> a_ug (levels, decimal)
cbo_forecasts = pd.read_csv(clean_data / 'master_projections_cleaned.csv')
cbo_forecasts['year'] = pd.to_datetime(cbo_forecasts['date']).dt.year
cbo_forecasts = (cbo_forecasts.groupby('year')['g (cbo baseline)']
                 .mean().reset_index())
cbo_forecasts = cbo_forecasts[(cbo_forecasts['year'] >= 2025) &
                              (cbo_forecasts['year'] <= 2035)]
a_ug = (cbo_forecasts['g (cbo baseline)'].values) / 100.0

# read in higher TFP growth scenario from CBO updated values
cbo_ai = pd.read_excel(clean_data / 'cbo_ai_projections.xlsx', sheet_name='higher_tfp_data')
cbo_ai = cbo_ai[(cbo_ai['year'] >= 2025) &
                              (cbo_ai['year'] <= 2035)]
# now estimate percent change in real GDP
a_ug_ai = (cbo_ai['g (cbo ai)'].values) / 100.0

# Senate TBL baseline for s -> a_s (levels, decimal)
forecasts = pd.read_csv(clean_data / 'master_projections_cleaned.csv')
forecasts['year'] = pd.to_datetime(forecasts['date']).dt.year
a_s_alternative = (forecasts.groupby('year')['s (cbo baseline)']
             .mean().reset_index())
a_s_alternative = a_s_alternative[(a_s_alternative['year'] >= 2025) &
                      (a_s_alternative['year'] <= 2035)]
a_s_alternative = (a_s_alternative['s (cbo baseline)'].values) / 100.0
forecasts = (forecasts.groupby('year')['s (tbl senate, permanent)']
             .mean().reset_index())
forecasts = forecasts[(forecasts['year'] >= 2025) &
                      (forecasts['year'] <= 2035)]
a_s = (forecasts['s (tbl senate, permanent)'].values) / 100.0

# -------------------------------------------------
# Core simulator (FIRST-DIFFERENCE r law)
# -------------------------------------------------
def simulate_scenario(
    c_val,
    a_s_vec, a_ug,
    r_star, beta_r, rho, sigma,
    s_g, s_x, s_r, s_s,
    x0=0.0, r0=None, b0=0.9656,
    n_years=10, n_simulations=5000, label=""
):
    """
    Fully-enriched SDSA:
      - g_t = a_ug[t] + x_t + e_g
      - x_t random walk
      - r_t uses FIRST-DIFFERENCE law:
            Δr_t = β_r(Δb_{t-1} - ρΔb_{t-2}) + ρΔr_{t-1} + η_t,  η_t = ε_t - ε_{t-1}
        with r_t = r_{t-1} + Δr_t
      - r_av,t = σ r_av,t-1 + (1-σ) r_t
      - s_t = (1-c) a_s[t] + c (r_av,t - g_t) b_{t-1} + e_s
      - b_t = b_{t-1} + ((r_av,t - g_t)/(1+g_t)) b_{t-1} - s_t
    """
    results = []

    for sim in range(n_simulations):
        # state arrays
        x   = np.zeros(n_years)
        g   = np.zeros(n_years)
        r   = np.zeros(n_years)
        r_av= np.zeros(n_years)
        s   = np.zeros(n_years)
        b   = np.zeros(n_years)

        # shocks (draw on the fly; we only need last ε for η_t)
        eps_r_prev = 0.0

        # initials
        x[0]   = x0
        g[0]   = a_ug[0] + x[0] + np.random.normal(0, s_g)
        r[0]   = r0 if r0 is not None else r_star
        r_av[0]= r[0]
        b[0]   = b0
        s[0]   = (1 - c_val) * a_s_vec[0] + c_val * (r_av[0] - g[0]) * b[0] + np.random.normal(0, s_s)

        # update b[1] once we have g[0], r_av[0], s[0]
        # (note: main loop starts at t=1, so b[1] will be computed there)

        for t in range(1, n_years):
            # shocks
            e_g = np.random.normal(0, s_g)
            e_x = np.random.normal(0, s_x)
            eps_r = np.random.normal(0, s_r)   # ε_t for r
            e_s = np.random.normal(0, s_s)

            # x, g
            x[t] = x[t-1] + e_x
            g[t] = a_ug[t] + x[t] + e_g

            # compute debt BEFORE r update? we need Δb_{t-1} and Δb_{t-2} for r
            # ensure b[t-1] is already set (it is from previous iteration)
            # build deltas for r law
            db_t_1 = (b[t-1] - b[t-2]) if t >= 2 else 0.0       # Δb_{t-1}
            db_t_2 = (b[t-2] - b[t-3]) if t >= 3 else 0.0       # Δb_{t-2}
            dr_t_1 = (r[t-1] - r[t-2]) if t >= 2 else 0.0       # Δr_{t-1}
            eta_t  = eps_r - eps_r_prev                          # MA(1) innovation

            # FIRST-DIFFERENCE r update
            dr_t = beta_r * (db_t_1 - rho * db_t_2) + rho * dr_t_1 + eta_t
            r[t] = r[t-1] + dr_t

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
b0      = 0.9656
r_star  = 0.01
r0      = 0.04

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
def _band_by_year(df, var):
        g = (df.groupby('year')[var]
             .agg(median='median',
                  p25=lambda x: np.percentile(x,25),
                  p75=lambda x: np.percentile(x,75))
             .reset_index())
        return g

def summarize_and_plot(sim_results_by_regime, graphics_path: Path, d_dict: dict, r_star: float, b0: float):
    graphics_path.mkdir(parents=True, exist_ok=True)

    # add derived vars and collect y-lims across ALL regimes
    ylim_store = {k: [] for k in ['g','r_av','rg','b','slope','curvature','interest_share']}
    enriched = {}

    for label, df_sim in sim_results_by_regime.items():
        df_sim = df_sim.copy()
        df_sim['rg'] = df_sim['r'] - df_sim['g']
        df_sim['slope'] = df_sim.groupby('sim', group_keys=False)['b'].apply(
            lambda s: _sg_deriv(s, window=5, poly=2, deriv=1))
        df_sim['curvature'] = df_sim.groupby('sim', group_keys=False)['b'].apply(
            lambda s: _sg_deriv(s, window=5, poly=2, deriv=2))
        df_sim['interest_share'] = df_sim['r_av'] * df_sim.groupby('sim')['b'].shift(1).fillna(b0)

        for var in ylim_store:
            ylim_store[var].extend(df_sim[var].values)

        enriched[label] = df_sim

    # global axis ranges (robust to tails)
    ylim_bounds = {k: (np.percentile(v, 0.5), np.percentile(v, 99.5)) for k, v in ylim_store.items()}

    # overlay plots for each metric
    def _plot_overlay(var, title, ylab, yline=None, ylim_key=None, fname=None):
        plt.figure(figsize=(11,7))
        for label, c_val in d_dict.items():
            df_sim = enriched[label]
            var_name = var if var != 'r' else 'r_av'  # we plot r_av as "r"
            g = _band_by_year(df_sim, var_name)
            plt.plot(g['year'], g['median'], label=f"{label.title()} (c={c_val:.2f})")
            plt.fill_between(g['year'], g['p25'], g['p75'], alpha=0.20)
        if yline is not None:
            plt.axhline(y=yline, color='black', linestyle='--', linewidth=0.9)
        plt.title(title)
        plt.xlabel('Year'); plt.ylabel(ylab)
        plt.grid(True)
        if ylim_key:
            lo, hi = ylim_bounds[ylim_key]
            plt.ylim(lo, hi)
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
                    range(2025, 2025 + n_years),
                    df_sim_i['g'],
                    color='green', alpha=0.2,
                    label='simulations w/ > 2.8% avg growth' if not high_labeled else ""
                )
                high_labeled = True
            elif mean_g < 0.0075:
                plt.plot(
                    range(2025, 2025 + n_years),
                    df_sim_i['g'],
                    color='red', alpha=0.2,
                    label='simulations w/ < 0.75% avg growth' if not low_labeled else ""
                )
                low_labeled = True
            else:
                plt.plot(
                    range(2025, 2025 + n_years),
                    df_sim_i['g'],
                    color='gray', alpha=0.01,
                    label='all other simulations' if not other_labeled else ""
                )
                other_labeled = True
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.9)
    plt.title('Growth (g) Rates: Median — Responsible (c=0.15) Only')
    plt.xlabel('Year')
    plt.ylabel('real growth rate (g)')
    plt.grid(True)
    plt.legend(loc='best', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(graphics_path / 'sdsa_enrichment1_g_rates_responsible_only.pdf', dpi=300)
    plt.show()

    _plot_overlay('r',   # uses r_av internally
                  'Interest Rate (r_av): Median & IQR — Regimes',
                  'r_av',
                  yline=r_star,
                  ylim_key='r_av',
                  fname='sdsa_enrichment1_r_av_overlay.pdf')

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

    _plot_overlay('interest_share',
                  'Interest Share (r_av * b[-1]): Median & IQR — Regimes',
                  'Interest Share',
                  yline=0.02,
                  ylim_key='interest_share',
                  fname='sdsa_enrichment1_interest_share_overlay.pdf')

    _plot_overlay('slope',
                  'Slope of Debt (b): Median & IQR — Regimes',
                  'Slope',
                  yline=0.0,
                  ylim_key='slope',
                  fname='sdsa_enrichment1_slope_overlay.pdf')

    _plot_overlay('curvature',
                  'Curvature of Debt (b): Median & IQR — Regimes',
                  'Curvature',
                  yline=0.0,
                  ylim_key='curvature',
                  fname='sdsa_enrichment1_curvature_overlay.pdf')

    return enriched  # returns enriched frames in case you want to export

# ---- call it (note we pass d_dict and r_star/b0 so legends/lines show correctly)
sim_results_by_regime = {}
for label, c_val in d_dict.items():
    for beta_label, beta_r in beta_r_dict.items():
        df_sim = simulate_scenario(
            c_val=c_val, a_s_vec=a_s, a_ug=a_ug,
            r_star=r_star, beta_r=beta_r, rho=rho, sigma=sigma,
            s_g=s_g, s_x=s_x, s_r=s_r, s_s=s_s,
            x0=0.0, r0=r0, b0=b0,
            n_years=n_years, n_simulations=n_sims,
            label=f"{label} (β_r={beta_label})"
        )
        sim_results_by_regime[label] = df_sim  # one β shown here

# overlay plots with both regimes + c in legend text
enriched_frames = summarize_and_plot(sim_results_by_regime, output, d_dict, r_star, b0)

# (optional) export
all_sim_results = pd.concat(enriched_frames.values(), ignore_index=True)
all_sim_results.to_csv(output / 'sdsa_enrichment1_sim_results.csv', index=False)

# -------------------------------------------------
# Addendum - AI Boom Scenarios (0.5pp and 1pp Productivity Boost)
# -------------------------------------------------

# Each scenario provides AI GDP paths, growth rates, and changes in primary deficit (% of GDP)
# Baseline years 2025–2035
years = np.arange(2025, 2036)

# --- 0.5 pp productivity boost ---
a_ug_ai_05 = np.array([
    2.125, 1.938, 1.988, 2.088, 2.250,
    2.411, 2.429, 2.466, 2.466, 2.451, 2.428
]) / 100
change_primary_deficit_pct_gdp_05 = np.array([
    0.0, -0.019924279, -0.064984957, -0.137927756, -0.246002998,
    -0.389646498, -0.545737236, -0.710450817, -0.885442132,
    -1.068350832, -1.262290796
]) / 100  # convert percentage points to decimals

# --- 1 pp productivity boost ---
a_ug_ai_10 = np.array([
    2.125, 2.932, 2.906, 2.923, 2.997,
    3.064, 3.093, 3.141, 3.150, 3.145, 3.131
]) / 100
change_primary_deficit_pct_gdp_10 = np.array([
    0.0, -0.182794733, -0.445974749, -0.719954032,
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
    "Responsible (c=0.15)": 0.15
}

# Run simulations for all combinations
sim_results_by_ai = {}
for ai_label, params in ai_scenarios.items():
    for c_label, c_val in c_scenarios.items():
        df_sim = simulate_scenario(
            c_val=c_val, a_s_vec=params["a_s"], a_ug=params["a_ug"],
            r_star=r_star, beta_r=beta_r_dict["3 bps"], rho=rho, sigma=sigma,
            s_g=s_g, s_x=s_x, s_r=s_r, s_s=s_s,
            x0=0.0, r0=r0, b0=b0,
            n_years=n_years, n_simulations=n_sims,
            label=f"{ai_label} - {c_label}"
        )
        sim_results_by_ai[f"{ai_label} - {c_label}"] = df_sim

# ---- Plot overlays: Debt paths ----
for c_label, c_val in c_scenarios.items():
    filtered = {k: v for k, v in sim_results_by_ai.items() if c_label in k}

    plt.figure(figsize=(11,7))
    for label, df_sim in filtered.items():
        g = _band_by_year(df_sim, "b")
        plt.plot(g["year"], g["median"], label=label)
        plt.fill_between(g["year"], g["p25"], g["p75"], alpha=0.20)
    plt.axhline(y=b0, color="black", linestyle="--", linewidth=0.9)
    plt.ylim(b0 - 0.1, 1.7)
    plt.title(f"Debt (b): Median & IQR — AI Boom Scenarios vs. Baseline ({c_label})")
    plt.xlabel("Year"); plt.ylabel("Debt-to-GDP Ratio (b)")
    plt.grid(True)
    plt.legend(loc="best", fontsize="x-large")
    plt.tight_layout()
    plt.savefig(output / f"sdsa_enrichment1_b_overlay_ai_booms_{c_val:.2f}.pdf", dpi=300)
    plt.show()

# ---- Optional overlay for growth ----
plt.figure(figsize=(11,7))
for ai_label, params in ai_scenarios.items():
    plt.plot(years, params["a_ug"], label=ai_label)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.9)
plt.title("Mean Growth Rate Paths: Baseline vs. AI Productivity Scenarios")
plt.xlabel("Year"); plt.ylabel("a_ug (growth rate)")
plt.grid(True)
plt.legend(fontsize="x-large")
plt.tight_layout()
plt.savefig(output / "sdsa_enrichment1_a_ug_paths_ai_booms.pdf", dpi=300)
plt.show()

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
    "AI Boom (1.0pp Productivity) - Irresponsible (c=0.00)",
    "CBO Baseline - Responsible (c=0.15)",
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

# 2. Plot historical vs simulated distributions (transparent histograms)
plt.figure(figsize=(8,5))
sns.histplot(
    net_interest_series,
    label="Historical",
    stat="density",
    element="step",     # outline only
    fill=False,
    linewidth=2.5,
    alpha=1
)

graphing_labels = {
    "CBO Baseline - Irresponsible (c=0.00)": "No AI Boom - Baseline",
    "CBO Baseline - Responsible (c=0.15)": "No AI Boom - Responsible",
    "AI Boom (1.0pp Productivity) - Irresponsible (c=0.00)": "AI Boom - 1.0pp Productivity, Baseline"
}

for ai_scenario in target_labels:
    if ai_scenario not in sim_results_by_ai:
        continue
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
        alpha=1, linestyle='--'
    )

    if ai_scenario == "AI Boom (1.0pp Productivity) - Irresponsible (c=0.00)":
        # print percentage of simulations above max of historical series
        max_historic = net_interest_series.max()
        pct_above = (sim_df['net_interest_sim'] > max_historic).mean() * 100
        print(f"Percentage of simulations with net interest > {max_historic:.2f}%: {pct_above:.2f}%")

plt.title("Interest Payments as % of GDP: Historical vs Simulated")
plt.xlabel("Interest (% of GDP)")
plt.ylabel("Density (Relative Frequency)")
plt.legend()
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
plt.title("Effective Interest Rate on Debt: Historical vs Simulated")
plt.xlabel("Effective Interest Rate (r_av)")
plt.ylabel("Density (Relative Frequency)")
plt.legend()
plt.tight_layout()
plt.show()

# now compare the debt accumulation snowball term (r^av - g) / (1 + g) * b[-1] to this same term applied historically
net_interest_df['debt_snowball_term'] = (
    (net_interest_df['effective_r'] - net_interest_df['real_gdp_growth_rate']) /
    (1 + net_interest_df['real_gdp_growth_rate'] / 100)
) * (net_interest_df['debt_held_by_public'] / 100)
plt.figure(figsize=(8,5))
sns.histplot(
    net_interest_df['debt_snowball_term'],
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
plt.title("Debt Accumulation Snowball Term: Historical vs Simulated")
plt.xlabel("Debt Snowball Term")
plt.ylabel("Density (Relative Frequency)")
plt.legend()
plt.tight_layout()
plt.show()

# now compare r_av - g directly
plt.figure(figsize=(8,5))
sns.histplot(
    net_interest_df['effective_r'] - net_interest_df['real_gdp_growth_rate'],
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
plt.title("Interest-Growth Differential (r_av - g): Historical vs Simulated")
plt.xlabel("r_av - g")
plt.ylabel("Density (Relative Frequency)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Addendum - Using pre-OBBBA CBO s projections
# -------------------------------------------------
# fix c = 0.075
c = 0.075 
# run sims and plot overlay -- ONLY for debt, comparing a_s to a_s_alternative 
sim_results_s_paths = {}
for a_s_label, a_s_vec in zip(['Post-OBBBA TBL Projected Deficit', 'Pre-OBBBA CBO Baseline Projected Deficit'], 
                              [a_s, a_s_alternative]):
    df_sim = simulate_scenario(
        c_val=c, a_s_vec=a_s_vec, a_ug=a_ug,
        r_star=r_star, beta_r=beta_r_dict['3 bps'], rho=rho, sigma=sigma,
        s_g=s_g, s_x=s_x, s_r=s_r, s_s=s_s,
        x0=0.0, r0=r0, b0=b0,
        n_years=n_years, n_simulations=n_sims,
        label=f"{a_s_label} (c={c:.3f})"
    )
    sim_results_s_paths[a_s_label] = df_sim
# plot only debt path overlay by hand, no function
color_dict = {
    'Post-OBBBA TBL Projected Deficit': 'blue',
    'Pre-OBBBA CBO Baseline Projected Deficit': 'orange'
}
plt.figure(figsize=(11,7))
for label, df_sim in sim_results_s_paths.items():
    g = _band_by_year(df_sim, 'b').reset_index()
    # Convert "year" to calendar year for both median and simulations
    g['calendar_year'] = 2024 + g['year']  # so that year=1 → 2025, etc.
    # Plot median
    plt.plot(g['calendar_year'], g['median'], label=label, linewidth=2.5, color=color_dict[label])
    # Optional fill between p25/p75
    plt.fill_between(g['calendar_year'], g['p25'], g['p75'], alpha=0.20, color=color_dict[label])
    # Plot all simulation paths in background
    '''for sim_id, df_sim_i in df_sim.groupby('sim'):
        plt.plot(
            2024 + df_sim_i['year'],
            df_sim_i['b'],
            color=color_dict[label], alpha=0.01
        )'''
plt.axhline(y=b0, color='black', linestyle='--', linewidth=0.9)
plt.ylim(b0 - 0.1, 1.7)
plt.title(f'Debt (b): Median & IQR — Senate TBL vs. CBO Baseline Savings')
plt.xlabel('Year')
plt.ylabel('Debt-to-GDP ratio (b)')
plt.grid(True)
plt.legend(loc='best', fontsize='x-large')
plt.tight_layout()
plt.savefig(output / 'sdsa_enrichment1_b_overlay_s_paths.pdf', dpi=300)
plt.show()