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

# Senate TBL baseline for s -> a_s (levels, decimal)
forecasts = pd.read_csv(clean_data / 'master_projections_cleaned.csv')
forecasts['year'] = pd.to_datetime(forecasts['date']).dt.year
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
rho   = 0.85   # AR coefficient in Δr law (on Δr_{t-1})
sigma = 0.80   # pass-through to r_av

# fiscal reaction regimes (only enrichment runs)
d_dict = {
    'irresponsible': 0.00,
    'responsible'  : 0.15,
    'very responsible': 0.30
}

# β_r in pp per 1-pp debt (3 bps baseline)
beta_r_dict = {'3 bps': 0.01}

# -------------------------------------------------
# Simulate, summarize, and plot (Enrichment 1 only)
# -------------------------------------------------
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

    # helper: compute median & IQR by year
    def _band_by_year(df, var):
        g = (df.groupby('year')[var]
             .agg(median='median',
                  p25=lambda x: np.percentile(x,25),
                  p75=lambda x: np.percentile(x,75))
             .reset_index())
        return g

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