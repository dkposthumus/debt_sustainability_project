# ============================================
# ENRICHMENT 1 / FULLY-ENRICHED SDSA SIMULATOR
# with FIRST-DIFFERENCE r law of motion
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from fredapi import Fred
import statsmodels.api as sm
import seaborn as sns
from scipy.signal import savgol_filter

# -------------------------------------------------
# Paths, plotting style, and helpers
# -------------------------------------------------
work_dir = Path(__file__).resolve().parent.parent
raw_data = work_dir / 'data' / 'raw'
clean_data = work_dir / 'data' / 'clean'
output = work_dir / 'output' / 'figures'
output.mkdir(parents=True, exist_ok=True)

plt.style.use(str(Path(__file__).resolve().parent / 'mahoney_lab.mplstyle'))

plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'figure.titlesize': 26,
})

fred = Fred(api_key=(work_dir / 'fred_api_key.txt').read_text().strip())

def get_fred_series(series_id, series_name, retries=4, backoff=1.5):
    import time
    last_err = None
    for attempt in range(retries):
        try:
            s = fred.get_series(series_id)
            break
        except (ValueError, Exception) as e:
            last_err = e
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    df = pd.DataFrame(s, columns=[series_name])
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    return df

def _sg_deriv(series: pd.Series, window: int = 5, poly: int = 2, deriv: int = 1) -> pd.Series:
    """
    Savitzky–Golay derivative of a 1D pandas Series.
    Returns a Series aligned to the input index.

    Notes:
      - If the series is shorter than the required window, returns all-NaN.
      - Ensures window is odd and <= len(series).
      - Uses delta=1.0 (your time step is 1 year in the simulation arrays).
    """
    s = pd.Series(series)  # ensure Series
    n = len(s)
    if n == 0:
        return s.copy()
    # choose an odd window <= n
    w = int(window)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        w = 3
    if w > n:
        w = n if (n % 2 == 1) else (n - 1)
    # if still not feasible, return NaNs
    if w < (poly + 2) or w < 3 or w > n:
        return pd.Series(np.nan, index=s.index)
    x = s.to_numpy(dtype=float)
    try:
        y = savgol_filter(x, window_length=w, polyorder=poly, deriv=deriv, delta=1.0, mode="interp")
    except Exception:
        return pd.Series(np.nan, index=s.index)
    return pd.Series(y, index=s.index)

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

# -------------------------------------------------
# Configuration: year ranges and initial conditions
# -------------------------------------------------
HIST_START  = 2000     # start of historical debt series
HIST_END    = 2025     # last year of observed data
PROJ_START  = 2026     # first year of CBO projection window
PROJ_END    = 2036     # last year of CBO projection window (Feb 2026 outlook covers 2026-2036)
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

# CBO baseline primary balance -> a_s (levels, decimal). The Feb 2026 CBO outlook
# already incorporates OBBBA, so we plug it in directly without any post-hoc adjustment.
a_s_df = (master_proj[proj_mask].groupby('year')['s (cbo baseline)']
          .mean().reset_index())
a_s = (a_s_df['s (cbo baseline)'].values) / 100.0

debt_hist = get_fred_series('FYGFGDQ188S', 'debt_pct_gdp')
debt_hist = debt_hist[debt_hist['date'] >= f'{HIST_START}-01-01'].copy()
debt_hist['b_hist'] = debt_hist['debt_pct_gdp'] / 100.0
debt_hist = debt_hist[debt_hist['date'].dt.month == 10].copy()
debt_hist['calendar_year'] = debt_hist['date'].dt.year
debt_hist = debt_hist[(debt_hist['calendar_year'] >= HIST_START) &
                      (debt_hist['calendar_year'] <= HIST_END)][['calendar_year', 'b_hist']].dropna()

# Initial debt level (start of simulation): most recent observed Q4 debt-to-GDP ratio.
b0 = float(debt_hist.loc[debt_hist['calendar_year'] == HIST_END, 'b_hist'].iloc[0])

# Initial average effective REAL rate on existing debt at sim start.
# Nominal rate: annualized net federal interest (interest paid - interest received)
#               / debt held by public.
# Deflator:    10-year breakeven inflation (T10YIE), the market-implied expected inflation
#              over the next 10 years. We use this rather than realized GDPDEF YoY because
#              r_av is the rate paid on a portfolio of mostly multi-year debt, so the
#              relevant inflation horizon is forward-looking and long-term, not last quarter's
#              realized print.
int_paid     = get_fred_series('A091RC1Q027SBEA', 'int_paid')      # SAAR, $bn, quarterly
int_received = get_fred_series('B094RC1Q027SBEA', 'int_received')  # SAAR, $bn, quarterly
debt_dollars = get_fred_series('FYGFDPUN', 'debt_mln')             # end-of-month, $mn
breakeven    = get_fred_series('T10YIE', 'breakeven')              # 10y breakeven, daily, %

for d in (int_paid, int_received, debt_dollars, breakeven):
    d['date'] = d['date'].dt.to_period('Q').dt.to_timestamp()
debt_dollars = debt_dollars.groupby('date', as_index=False)['debt_mln'].last()
breakeven    = breakeven.groupby('date', as_index=False)['breakeven'].mean()

rav_df = (int_paid.merge(int_received, on='date')
                  .merge(debt_dollars, on='date')
                  .merge(breakeven, on='date'))
rav_df['nominal_rate'] = (rav_df['int_paid'] - rav_df['int_received']) / (rav_df['debt_mln'] / 1000.0)
rav_df['expected_infl'] = rav_df['breakeven'] / 100.0
rav_df = rav_df.dropna()

target_year_obs = rav_df[rav_df['date'].dt.year == HIST_END]
if target_year_obs.empty:
    target_year_obs = rav_df.tail(4)  # fall back: last 4 quarters
r_av0 = float((target_year_obs['nominal_rate'] - target_year_obs['expected_infl']).mean())

growth = get_fred_series('A191RL1Q225SBEA', 'gdp_growth_rate') # quarterly, percent (real)
interest = get_fred_series('REAINTRATREARAT10Y', 'interest_rate') # monthly, percent (real)
snowball_hist = growth.merge(interest, on='date', how='outer')
snowball_hist = snowball_hist[snowball_hist['date'].dt.month == 10].copy()
snowball_hist['calendar_year'] = snowball_hist['date'].dt.year
snowball_hist = snowball_hist[(snowball_hist['calendar_year'] >= HIST_START) &
                                (snowball_hist['calendar_year'] <= HIST_END)].copy()
snowball_hist = debt_hist.merge(snowball_hist, on='calendar_year', how='left')
# Note: historical r-g uses the market 10yr real rate as a proxy since historical r_av is not observed
snowball_hist['rg_hist'] = (snowball_hist['interest_rate'] - snowball_hist['gdp_growth_rate']) / 100.0
snowball_hist['snowball'] = ((snowball_hist['interest_rate'] - snowball_hist['gdp_growth_rate']) / 100.0
                             / (1.0 + snowball_hist['gdp_growth_rate'] / 100.0)
                             * snowball_hist['b_hist'])

# -------------------------------------------------
# Core simulator (CBO-baseline r with debt feedback)
# -------------------------------------------------
def compute_cbo_baseline_debt(a_ug, a_ur, a_s_vec, sigma, r_av0, b0, n_years):
    """
    Deterministic CBO baseline debt path (no shocks, no fiscal feedback, c=0).
    Returns db_cbo: array of length n_years where db_cbo[t] = b_cbo[t] - b_cbo[t-1].
    This is what CBO's rate path a_ur already prices in.
    """
    b_cbo = np.zeros(n_years)
    db_cbo = np.zeros(n_years)
    r_av_cbo = np.zeros(n_years)

    b_prev = b0
    r_av_prev = r_av0 if r_av0 is not None else a_ur[0]

    for t in range(n_years):
        r_av_cbo[t] = sigma * r_av_prev + (1.0 - sigma) * a_ur[t]
        g_t = a_ug[t]
        s_t = a_s_vec[t]  # no feedback, no shocks
        b_cbo[t] = b_prev + ((r_av_cbo[t] - g_t) / (1.0 + g_t)) * b_prev - s_t
        db_cbo[t] = b_cbo[t] - b_prev
        b_prev = b_cbo[t]
        r_av_prev = r_av_cbo[t]

    return db_cbo


def simulate_scenario(
    c_val,
    a_s_vec, a_ug, a_ur, db_cbo,
    beta_r, rho, sigma,
    s_g, s_x, s_r, s_s,
    x0=0.0, r_av0=None, b0=None,
    n_years=10, n_simulations=20000, label="", seed=42,
):
    """
    Fully-enriched SDSA:
      - g_t = a_ug[t] + x_t + e_g           (CBO growth baseline + random walk + shock)
      - x_t random walk
      - r_t = a_ur[t] + z_t                  (CBO rate baseline + deviation)
        where z_t responds to SURPRISE debt changes (sim minus CBO baseline):
            Δz_t = β_r(Δb^s_{t-1} - ρΔb^s_{t-2}) + ρΔz_{t-1} + η_t
            Δb^s_t = Δb_t - Δb^cbo_t
      - r_av,t = σ r_av,t-1 + (1-σ) r_t     (pass-through to avg rate on debt)
      - s_t = (1-c) a_s[t] + c ((r_av,t - g_t)/(1+g_t)) b_{t-1} + e_s
      - b_t = b_{t-1} + ((r_av,t - g_t)/(1+g_t)) b_{t-1} - s_t
    """
    np.random.seed(seed)
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
        s[0]   = (1 - c_val) * a_s_vec[0] + c_val * ((r_av[0] - g[0]) / (1.0 + g[0])) * b0 + np.random.normal(0, s_s)
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

            # surprise debt changes (sim minus CBO baseline)
            db_sim_t_1 = (b[t-1] - b[t-2]) if t >= 2 else (b[0] - b0)
            db_sim_t_2 = (b[t-2] - b[t-3]) if t >= 3 else (b[0] - b0) if t == 2 else 0.0

            db_surprise_t_1 = db_sim_t_1 - db_cbo[t-1]
            db_surprise_t_2 = db_sim_t_2 - db_cbo[t-2] if t >= 2 else 0.0

            dz_t_1 = (z[t-1] - z[t-2]) if t >= 2 else 0.0
            eta_t  = eps_r - eps_r_prev

            # first-difference law on deviations, driven by surprise debt
            dz_t = beta_r * (db_surprise_t_1 - rho * db_surprise_t_2) + rho * dz_t_1 + eta_t
            z[t] = z[t-1] + dz_t
            r[t] = a_ur[t] + z[t]

            # pass-through to average rate paid on debt
            r_av[t] = sigma * r_av[t-1] + (1.0 - sigma) * r[t]

            # primary balance with fiscal feedback
            s[t] = (1 - c_val) * a_s_vec[t] + c_val * ((r_av[t] - g[t])/(1+g[t])) * b[t-1] + e_s

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
# horizons and baseline levels
n_years = len(a_ug)
n_sims  = 20000
SIM_SEED = 42  # passed into simulate_scenario; see seeding logic there

# stochastic vols
s_g = 0.005
s_x = 0.002
s_r = 0.005
s_s = 0.010

# smoothing / persistence
sigma = 0.80   # pass-through to r_av

# β_r and ρ from the term premium regression in 02_estimate_term_premium.py
_calib = pd.read_csv(clean_data / 'term_premium_calibration.csv').set_index('parameter')['estimate']
beta_r = float(_calib['beta_r'])
rho    = float(_calib['rho'])
beta_r_dict = {f'{beta_r*100:.2f} bps': beta_r}

# fiscal reaction regimes (only enrichment runs)
d_dict = {
    'irresponsible': 0.00,
    'responsible'  : 0.15,
    'very responsible': 0.30
}

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
    hist_anchor = snowball_hist[snowball_hist['calendar_year'] == anchor_year]
    anchor_rg = (hist_anchor['interest_rate'].values[0] - hist_anchor['gdp_growth_rate'].values[0]) / 100.0 if len(hist_anchor) > 0 else 0.0
    anchor_snowball = hist_anchor['snowball'].values[0] if len(hist_anchor) > 0 else 0.0
    anchor_g = hist_anchor['gdp_growth_rate'].values[0] / 100.0 if len(hist_anchor) > 0 else 0.0

    ylim_store = {k: [] for k in ['g','r_av','rg','b','slope','curvature','interest_share','snowball']}
    enriched = {}

    for label, df_sim in sim_results_by_regime.items():
        df_sim = df_sim.copy()
        df_sim["calendar_year"] = sim_start_year + df_sim["year"] - 1
        df_sim['rg'] = df_sim['r_av'] - df_sim['g']
        df_sim['slope'] = df_sim.groupby('sim', group_keys=False)['b'].apply(
            lambda s: _sg_deriv(s, window=5, poly=2, deriv=1))
        df_sim['curvature'] = df_sim.groupby('sim', group_keys=False)['b'].apply(
            lambda s: _sg_deriv(s, window=5, poly=2, deriv=2))
        df_sim['interest_share'] = df_sim['r_av'] * df_sim.groupby('sim')['b'].shift(1).fillna(b0)
        df_sim['snowball'] = ((df_sim['r_av'] - df_sim['g']) / (1.0 + df_sim['g'])) * df_sim.groupby('sim')['b'].shift(1).fillna(b0)

        # Prepend a 2025 anchor row for each sim so projected lines connect to history
        sims = df_sim['sim'].unique()
        anchor_rows = pd.DataFrame({
            'year': 0,
            'sim': sims,
            'b': b0,
            'r': anchor_rg + anchor_g,
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
                color="black", linewidth=3.0, label="Historical (2000-2025)"
            )
            plt.axvline(sim_start_year - 1, color="black", linestyle="--", linewidth=0.9, alpha=0.8,
                    label="_nolegend_")
        if var == 'rg':
            rg_clean = snowball_hist.dropna(subset=["rg_hist"])
            _, rg_trend = sm.tsa.filters.hpfilter(rg_clean["rg_hist"], lamb=100)
            plt.plot(
                snowball_hist["calendar_year"], snowball_hist["rg_hist"],
                color="black", linewidth=1.2, alpha=0.25, label="_nolegend_"
            )
            plt.plot(
                rg_clean["calendar_year"], rg_trend.values,
                color="black", linewidth=3.0, label="Historical HP trend (2000-2025)"
            )
            plt.axvline(sim_start_year - 1, color="black", linestyle="--", linewidth=0.9, alpha=0.8,
                    label="_nolegend_")
        if var == "snowball":
            sb_clean = snowball_hist.dropna(subset=["snowball"])
            _, sb_trend = sm.tsa.filters.hpfilter(sb_clean["snowball"], lamb=100)
            plt.plot(
                snowball_hist["calendar_year"], snowball_hist["snowball"],
                color="black", linewidth=1.2, alpha=0.25, label="_nolegend_"
            )
            plt.plot(
                sb_clean["calendar_year"], sb_trend.values,
                color="black", linewidth=3.0, label="Historical HP trend (2005–2025)"
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

        plt.xlabel('', fontsize=20); plt.ylabel(ylab, fontsize=20)
        plt.grid(True)

        if ylim_key:
            lo, hi = ylim_bounds[ylim_key]

        plt.legend(loc='best', fontsize=18)
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
    plt.title('', fontsize=24)
    plt.xlabel('', fontsize=20)
    plt.ylabel('real growth rate (g)', fontsize=20)
    plt.grid(True)
    plt.legend(loc='best', fontsize=18)
    plt.tight_layout()
    plt.savefig(graphics_path / 'sdsa_enrichment1_g_rates_responsible_only.pdf', dpi=300)
    plt.close()

    _plot_overlay('rg',
                  'Interest-Growth Differential (r - g)',
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
# CBO baseline debt path for surprise-debt feedback
db_cbo_baseline = compute_cbo_baseline_debt(a_ug, a_ur, a_s, sigma, r_av0, b0, n_years)

sim_results_by_regime = {}
for label, c_val in d_dict.items():
    for beta_label, beta_r in beta_r_dict.items():
        df_sim = simulate_scenario(
            c_val=c_val, a_s_vec=a_s, a_ug=a_ug, a_ur=a_ur, db_cbo=db_cbo_baseline,
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

def add_stabilizing_and_adjustment(df_sim: pd.DataFrame, b0: float) -> pd.DataFrame:
    """
    Adds:
      - b_lag  : b_{t-1}
      - s_stab : ((r_av - g) / (1 + g)) * b_{t-1}   [stabilizing primary surplus]
      - AA     : s_stab - s                         [required adjustment]
    Assumes df_sim contains: sim, year (1..T), b, r_av, g, s (all decimals).
    """
    df = df_sim.sort_values(["sim", "year"]).copy()
    df["b_lag"] = df.groupby("sim")["b"].shift(1).fillna(b0)

    # Stabilizing primary balance consistent with:
    # b_t = b_{t-1} + ((r_av - g)/(1+g))*b_{t-1} - s_t
    denom = (1.0 + df["g"])
    # Safety: if 1+g is ever ~0 (shouldn't happen with your calibration), avoid division blowups.
    denom = denom.where(np.abs(denom) > 1e-8, np.nan)

    df["s_stab"] = ((df["r_av"] - df["g"]) / denom) * df["b_lag"]
    df["AA"] = df["s_stab"] - df["s"]
    return df

def summarize_median(df: pd.DataFrame, var: str, time_col: str = "calendar_year") -> pd.DataFrame:
    return (df.groupby(time_col)[var]
            .median()
            .reset_index()
            .rename(columns={var: "median"}))

def plot_primary_balance_story_popouts(
    scenarios: dict[str, pd.DataFrame],
    s_hist: pd.DataFrame,
    output_path: Path,
    *,
    sim_start_year: int,
    b0: float,
    recession_df: pd.DataFrame | None = None,
    hist_start_year: int = 1984,
    focus_year: str | int = "end",     # "end" or int calendar year
    popout_years: int = 6,
    popout_xmax: int = 2037,
    shortened_recessions: bool = False,
    annotation_fontsize: int = 18,
    fname: str = "sdsa_primary_balance_story_popouts.pdf",
    right_margin: float = 0.72,
    improvement_horizon_years: int = 5,
    print_hist_improvement_stats: bool = True,
):
    """
    Main plot:
      - Historical primary balance line (starts at hist_start_year)
      - Recession bars (via your add_recession_bars)
      - For each scenario: median s (solid) and median s_stab (dashed), same color
      - Legend: only scenario names (colors) + historical entry
      - No arrows/annotations/wedge shading, empty title

    Pop-outs (stacked on right):
      - For each scenario: solid + dashed, AA arrow at focus year, AA text (black, larger)
      - Pop-out x-axis extends to popout_xmax (default 2037)
    """

    # ---- history prep (YEAR axis in main plot)
    hist = s_hist.sort_values("calendar_year").copy()
    if not {"calendar_year", "s_hist"}.issubset(hist.columns):
        raise KeyError("s_hist must have columns: ['calendar_year', 's_hist']")
    hist = hist[hist["calendar_year"] >= hist_start_year].copy()
    hist_y = hist["s_hist"] * 100.0  # decimal -> pp

    # ---- choose focus year based on first scenario
    first_key = next(iter(scenarios.keys()))
    df0 = scenarios[first_key].copy()
    if "calendar_year" not in df0.columns:
        df0["calendar_year"] = sim_start_year + df0["year"] - 1

    if focus_year == "end":
        focus_year_val = int(df0["calendar_year"].max())
    elif isinstance(focus_year, int):
        focus_year_val = focus_year
    else:
        raise ValueError("focus_year must be 'end' or an int calendar year.")

    # ---- figure and main axis
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    fig.subplots_adjust(right=right_margin)

    # recession bars (IMPORTANT: your helper expects datetime x-axis)
    # Our main axis is YEAR-based, so we convert spans to year-fractions for compatibility.
    # Easiest: create a temporary axis in datetime? Not worth it.
    # Instead: we replicate your logic but on year-fraction, using your helper is not feasible on year axis.
    #
    # Since you explicitly asked to use your helper, we switch the MAIN x-axis to datetime years (Jan 1 each year),
    # and plot history/scenarios against datetime. This keeps your helper working.
    #
    # We'll re-map year columns to datetimes.

    # ---- Build datetime x for historical
    hist_dt = pd.to_datetime(hist["calendar_year"].astype(int).astype(str) + "-01-01")
    ax.plot(hist_dt, hist_y, linewidth=2.5, color="black", label="_nolegend_")

    # recession bars
    if recession_df is not None:
        # ensure datetime
        if not np.issubdtype(recession_df["date"].dtype, np.datetime64):
            recession_df = recession_df.copy()
            recession_df["date"] = pd.to_datetime(recession_df["date"])
        add_recession_bars(ax, recession_df, shortened=shortened_recessions)

    # focus year vertical line (datetime)
    focus_dt = pd.to_datetime(f"{focus_year_val}-01-01")
    ax.axvline(focus_dt, color="black", linestyle=":", linewidth=1.2, alpha=0.7)

    # ---- scenario medians on main axis
    scenario_summaries = {}  # {label: (s_med_dt_pp, st_med_dt_pp, color)}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None) or ["C0","C1","C2","C3","C4","C5"]

    # Get the 2025 historical primary balance to anchor the solid s lines
    anchor_year = sim_start_year - 1
    s_hist_2025 = hist.loc[hist["calendar_year"] == anchor_year, "s_hist"]
    s_anchor_dec = float(s_hist_2025.iloc[0]) if len(s_hist_2025) > 0 else np.nan

    for i, (sc_label, df_sim) in enumerate(scenarios.items()):
        df = df_sim.copy()
        if "calendar_year" not in df.columns:
            df["calendar_year"] = sim_start_year + df["year"] - 1

        # Fix the anchor row's s value to the actual 2025 historical primary balance
        anchor_mask = df["calendar_year"] == anchor_year
        if anchor_mask.any() and not np.isnan(s_anchor_dec):
            df.loc[anchor_mask, "s"] = s_anchor_dec

        df = add_stabilizing_and_adjustment(df, b0=b0)

        s_med = summarize_median(df, "s", time_col="calendar_year")
        st_med = summarize_median(df, "s_stab", time_col="calendar_year")

        # convert to pp + datetime x
        s_med["median"] *= 100.0
        st_med["median"] *= 100.0
        s_med["date"] = pd.to_datetime(s_med["calendar_year"].astype(int).astype(str) + "-01-01")
        st_med["date"] = pd.to_datetime(st_med["calendar_year"].astype(int).astype(str) + "-01-01")

        color = color_cycle[i % len(color_cycle)]

        # Solid s line: includes 2025 anchor (connects to historical)
        ax.plot(s_med["date"], s_med["median"], linewidth=2.6, color=color, label="_nolegend_")
        # Dashed s_stab line: starts at 2026 only (no historical anchor)
        st_med_proj = st_med[st_med["calendar_year"] >= sim_start_year]
        ax.plot(st_med_proj["date"], st_med_proj["median"], linewidth=2.6, linestyle="--", color=color, label="_nolegend_")

        scenario_summaries[sc_label] = (s_med, st_med, color)

    if print_hist_improvement_stats:
        H = improvement_horizon_years

        # build a continuous annual series from hist_start_year onward
        hist_series = (s_hist[["calendar_year", "s_hist"]]
                       .dropna()
                       .sort_values("calendar_year")
                       .copy())
        hist_series = hist_series[hist_series["calendar_year"] >= hist_start_year]

        if hist_series.empty:
            print(f"[AA benchmark] No historical s_hist data available since {hist_start_year}.")
        else:
            year_max = int(hist_series["calendar_year"].max())
            full_years = np.arange(hist_start_year, year_max + 1)

            s_ann = (hist_series.set_index("calendar_year")["s_hist"]
                     .reindex(full_years))  # decimals

            # H-year improvement in pp (surplus positive => improvement is increase)
            delta_H_pp = (s_ann - s_ann.shift(H)) * 100.0
            valid_windows = delta_H_pp.dropna()

            def _windows_ge(threshold_pp: float) -> list[tuple[int, int, float]]:
                """
                Returns list of (start_year, end_year, improvement_pp) for H-year windows
                where improvement >= threshold_pp.
                end_year corresponds to the index year of valid_windows.
                """
                if valid_windows.empty:
                    return []
                hits = valid_windows[valid_windows >= threshold_pp]
                return [(int(y - H), int(y), float(hits.loc[y])) for y in hits.index]

            print(f"\n[AA benchmark | {H}-year improvements since {hist_start_year}]")
            for sc_label, (s_med, st_med, _color) in scenario_summaries.items():
                # median AA at focus year (pp)
                s_at = float(s_med.loc[s_med["calendar_year"] == focus_year_val, "median"].iloc[0])
                st_at = float(st_med.loc[st_med["calendar_year"] == focus_year_val, "median"].iloc[0])
                aa_pp = st_at - s_at

                windows = _windows_ge(aa_pp)
                nwin = int(valid_windows.shape[0]) if not valid_windows.empty else 0
                k = len(windows)

                if nwin == 0:
                    print(f"  - {sc_label}: AA({focus_year_val})={aa_pp:.1f} pp | no valid historical {H}-year windows.")
                    continue

                print(f"  - {sc_label}: AA({focus_year_val})={aa_pp:.1f} pp | {k}/{nwin} windows (>= AA)")

                # Print the windows (rolling, overlapping)
                if k == 0:
                    print("      (none)")
                else:
                    for (y0, y1, imp) in windows:
                        print(f"      {y0}–{y1}: +{imp:.2f} pp")
            print("")

    # ---- main cosmetics (empty title)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Primary balance (% of GDP; surplus positive)", fontsize=20)
    '''ax.set_title("Ahistorical Adjustment by Projected Level of Congressional Responsibility",
                 fontsize=24)  # per your request'''
    ax.grid(True, alpha=0.35)

    # legend: ONLY scenario colors + historical entry
    handles = [Line2D([0], [0], color="black", linewidth=2.5, 
                      label="Historical primary balance")]
    for sc_label, (_, _, color) in scenario_summaries.items():
        handles.append(Line2D([0], [0], color=color, linewidth=3.0, 
                              label=sc_label))
    ax.legend(handles=handles, loc="best", fontsize=18)

    # set x-limits nicely
    xmin = pd.to_datetime(f"{hist_start_year}-01-01")
    xmax = pd.to_datetime(f"{popout_xmax}-01-01")
    ax.set_xlim(xmin, xmax)

    # ---- pop-outs stacked on right (datetime x-axis, extends to popout_xmax)
    n = len(scenario_summaries)
    left = right_margin + 0.02
    width = 0.26
    top = 0.88
    bottom = 0.12
    gap = 0.02
    h = (top - bottom - gap * (n - 1)) / n

    x0_year = max(sim_start_year, focus_year_val - (popout_years - 1))
    x0_dt = pd.to_datetime(f"{x0_year}-01-01")
    xmax_dt = pd.to_datetime(f"{popout_xmax}-01-01")

    # ------------------------------------------------------------
    # PRE-PASS: compute a single shared y-limit across all popouts
    # ------------------------------------------------------------
    all_vals = []
    for sc_label, (s_med, st_med, _color) in scenario_summaries.items():
        s_win = s_med[(s_med["calendar_year"] >= x0_year) & (s_med["calendar_year"] <= focus_year_val)]["median"]
        st_win = st_med[(st_med["calendar_year"] >= x0_year) & (st_med["calendar_year"] <= focus_year_val)]["median"]

        # include focus-year endpoints explicitly
        s_at = float(s_med.loc[s_med["calendar_year"] == focus_year_val, "median"].iloc[0])
        st_at = float(st_med.loc[st_med["calendar_year"] == focus_year_val, "median"].iloc[0])

        all_vals.extend(s_win.dropna().tolist())
        all_vals.extend(st_win.dropna().tolist())
        all_vals.append(s_at)
        all_vals.append(st_at)

    if len(all_vals) == 0:
        raise ValueError("Could not compute popout y-limits (no values found).")

    y_global_min = float(np.min(all_vals))
    y_global_max = float(np.max(all_vals))

    # padded, but shared across all popouts
    y_range = y_global_max - y_global_min
    shared_pad = max(0.5, 0.15 * y_range if y_range > 0 else 0.5)
    ylo_shared = y_global_min - shared_pad
    yhi_shared = y_global_max + shared_pad

    # ------------------------------------------------------------
    # Pop-out loop (uses shared y-limits)
    # ------------------------------------------------------------
    for j, (sc_label, (s_med, st_med, color)) in enumerate(scenario_summaries.items()):
        y = top - (j + 1) * h - j * gap
        ax_in = fig.add_axes([left, y, width, h])

        s_win = s_med[(s_med["calendar_year"] >= x0_year) & (s_med["calendar_year"] <= focus_year_val)].copy()
        # s_stab starts at sim_start_year (2026), not the anchor year
        st_win = st_med[(st_med["calendar_year"] >= max(x0_year, sim_start_year)) & (st_med["calendar_year"] <= focus_year_val)].copy()

        ax_in.plot(s_win["date"], s_win["median"], color=color, linewidth=2.2)
        ax_in.plot(st_win["date"], st_win["median"], color=color, linestyle="--", linewidth=2.2)

        # AA arrow + big black annotation
        s_at = float(s_med.loc[s_med["calendar_year"] == focus_year_val, "median"].iloc[0])
        st_at = float(st_med.loc[st_med["calendar_year"] == focus_year_val, "median"].iloc[0])
        aa_pp = st_at - s_at

        ax_in.axvline(focus_dt, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_in.annotate(
            "",
            xy=(focus_dt, st_at),
            xytext=(focus_dt, s_at),
            arrowprops=dict(arrowstyle="<->", linewidth=2.0, color=color),
        )
        ax_in.text(
            focus_dt + pd.Timedelta(days=40),
            (s_at + st_at) / 2.0,
            f"{sc_label}\nAA({focus_year_val}) = {aa_pp:.1f} pp",
            va="center",
            fontsize=annotation_fontsize,
            color="black",
        )

        ax_in.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_in.grid(True, alpha=0.25)
        ax_in.set_xlim(x0_dt, xmax_dt)

        # shared y-limits across popouts
        ax_in.set_ylim(ylo_shared, yhi_shared)

        ax_in.tick_params(labelsize=13)
        if j < n - 1:
            ax_in.set_xticklabels([])

    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / fname, dpi=300, 
                bbox_inches="tight", pad_inches=0.10)
    plt.show()

# historical primary balance series
s_hist = pd.read_excel(raw_data / "r_g_historic_data.xlsx", sheet_name="master")
s_hist = s_hist[["year", "primary balance (pct of GDP)"]].copy()
s_hist["s_hist"] = s_hist["primary balance (pct of GDP)"] / 100.0
s_hist = s_hist.rename(columns={"year": "calendar_year"})
# Append 2025 from CBO if not already present
if HIST_END not in s_hist["calendar_year"].values:
    s_2025 = master_proj[master_proj["year"] == HIST_END]["s (cbo baseline)"].mean() / 100.0
    s_hist = pd.concat([s_hist, pd.DataFrame({
        "calendar_year": [HIST_END],
        "primary balance (pct of GDP)": [s_2025 * 100.0],
        "s_hist": [s_2025],
    })], ignore_index=True)

recession_df = get_fred_series("USREC", "recession")  # date: datetime, recession: 0/1

c_regimes = {
    f"Irresponsible (c={d_dict['irresponsible']:.2f})": enriched_frames["irresponsible"],
    f"Responsible (c={d_dict['responsible']:.2f})": enriched_frames["responsible"],
    f"Very responsible (c={d_dict['very responsible']:.2f})": enriched_frames["very responsible"],
}

plot_primary_balance_story_popouts(
    scenarios=c_regimes,
    s_hist=s_hist,
    output_path=output,
    sim_start_year=SIM_START,
    b0=b0,
    recession_df=recession_df,
    hist_start_year=1984,
    focus_year="end",
    popout_xmax=2037,
    annotation_fontsize=18,
    fname="sdsa_primary_balance_story_popouts_c_regimes.pdf",
    shortened_recessions=False,
    print_hist_improvement_stats=True,
)

# -------------------------------------------------
# Addendum - AI Boom Scenarios (0.5pp and 1pp Productivity Boost)
# -------------------------------------------------

years = np.arange(PROJ_START, PROJ_END + 1)

# AI scenarios are produced by script 01 from CBO user-specified-projections workbooks.
# Each row is one (scenario, year): a_ug_ai = real GDP growth (decimal),
# delta_s_primary = primary-surplus delta vs CBO baseline (decimal of GDP, surplus-positive).
ai_scen = pd.read_csv(clean_data / 'ai_scenarios.csv')
ai_scen = ai_scen[(ai_scen['year'] >= PROJ_START) & (ai_scen['year'] <= PROJ_END)]

# Pretty labels for the scenarios encoded in workbook filenames
ai_label_map = {
    '05pp': 'AI Boom (0.5pp Productivity)',
    '10pp': 'AI Boom (1.0pp Productivity)',
}

# Build the scenario dict from whatever the CSV actually contains
ai_scenarios = {"CBO Baseline": {"a_ug": a_ug, "a_s": a_s}}
for label, sub in ai_scen.groupby('label'):
    sub = sub.sort_values('year')
    if len(sub) != n_years:
        print(f"[ai_scenarios] WARNING: scenario '{label}' has {len(sub)} years "
              f"covering {PROJ_START}-{PROJ_END}; expected {n_years}. Skipping.")
        continue
    pretty = ai_label_map.get(label, f"AI Boom ({label})")
    ai_scenarios[pretty] = {
        "a_ug": sub['a_ug_ai'].values,
        "a_s":  a_s + sub['delta_s_primary'].values,  # surplus-positive addition
    }

missing_scenarios = [lbl for lbl in ai_label_map if lbl not in set(ai_scen['label'])]
if missing_scenarios:
    print(f"[ai_scenarios] NOTE: missing workbook(s) for scenarios: {missing_scenarios}. "
          f"Drop cbo_user_specified_productivity_<label>.xlsx into data/raw/ to add them.")

# Define fiscal responsibility cases
c_scenarios = {
    "Irresponsible (c=0.00)": 0.00,
    "Responsible (c=0.15)": 0.15,
    "Very Responsible (c=0.30)": 0.30
}

# Run simulations for all combinations
# Precompute db_cbo for each AI scenario (different a_ug and a_s paths)
db_cbo_by_ai = {}
for ai_label, params in ai_scenarios.items():
    db_cbo_by_ai[ai_label] = compute_cbo_baseline_debt(
        params["a_ug"], a_ur, params["a_s"], sigma, r_av0, b0, n_years
    )

sim_results_by_ai = {}
for ai_label, params in ai_scenarios.items():
    for c_label, c_val in c_scenarios.items():
        df_sim = simulate_scenario(
            c_val=c_val, a_s_vec=params["a_s"], a_ug=params["a_ug"], a_ur=a_ur,
            db_cbo=db_cbo_by_ai[ai_label],
            beta_r=beta_r, rho=rho, sigma=sigma,
            s_g=s_g, s_x=s_x, s_r=s_r, s_s=s_s,
            x0=0.0, r_av0=r_av0, b0=b0,
            n_years=n_years, n_simulations=n_sims,
            label=f"{ai_label} - {c_label}"
        )
        sim_results_by_ai[f"{ai_label} - {c_label}"] = df_sim

# ---- Enrich AI sim results with calendar_year and 2025 anchor ----
def _add_anchor_and_calendar(df_sim, sim_start=SIM_START, b0_val=b0,
                              r_av0_val=r_av0):
    """Add calendar_year mapping and prepend 2025 anchor row to each sim."""
    hist_anchor = snowball_hist[snowball_hist['calendar_year'] == sim_start - 1]
    a_rg = (hist_anchor['interest_rate'].values[0] - hist_anchor['gdp_growth_rate'].values[0]) / 100.0 if len(hist_anchor) > 0 else 0.0
    a_sb = hist_anchor['snowball'].values[0] if len(hist_anchor) > 0 else 0.0
    a_g = hist_anchor['gdp_growth_rate'].values[0] / 100.0 if len(hist_anchor) > 0 else 0.0

    df = df_sim.copy()
    df["calendar_year"] = sim_start + df["year"] - 1
    df['snowball'] = ((df['r_av'] - df['g']) / (1.0 + df['g'])) * df.groupby('sim')['b'].shift(1).fillna(b0_val)
    df['r_minus_g'] = df['r_av'] - df['g']
    sims = df['sim'].unique()
    anchor = pd.DataFrame({
        'year': 0, 'sim': sims,
        'b': b0_val, 'r_av': r_av0_val, 'g': a_g,
        'r': a_rg + a_g,
        's': 0.0, 'c': df['c'].iloc[0], 'label': df['label'].iloc[0],
        'calendar_year': sim_start - 1,
        'snowball': a_sb,
        'r_minus_g': a_rg,
    })
    return pd.concat([anchor, df], ignore_index=True)

# ---- Plot overlays: Debt, r-g, Snowball paths ----
for c_label, c_val in c_scenarios.items():
    filtered = {k: v for k, v in sim_results_by_ai.items() if c_label in k}

    # Debt overlay
    plt.figure(figsize=(11,7))
    plt.plot(debt_hist["calendar_year"], debt_hist["b_hist"],
             color="black", linewidth=3.0, label="Historical (FRED, 2000-2025)")
    for label, df_sim in filtered.items():
        df_a = _add_anchor_and_calendar(df_sim)
        g = _band_by_year(df_a, "b", time_col="calendar_year")
        plt.plot(g["time"], g["median"], label=label)
        plt.fill_between(g["time"], g["p25"], g["p75"], alpha=0.20)
    plt.axhline(y=b0, color="black", linestyle="--", linewidth=0.9)
    plt.xlim(debt_hist["calendar_year"].min(), SIM_START + n_years - 1)
    plt.ylim(b0 - 0.1, 1.7)
    plt.xlabel("", fontsize=20); plt.ylabel("Debt-to-GDP Ratio (b)", fontsize=20)
    plt.grid(True)
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout()
    plt.savefig(output / f"sdsa_enrichment1_b_overlay_ai_booms_{c_val:.2f}.pdf", dpi=300)
    plt.close()

    # r-g overlay
    plt.figure(figsize=(11,7))
    rg_clean = snowball_hist.dropna(subset=["rg_hist"])
    _, rg_trend = sm.tsa.filters.hpfilter(rg_clean["rg_hist"], lamb=100)
    plt.plot(snowball_hist["calendar_year"], snowball_hist["rg_hist"],
             color="black", linewidth=1.2, alpha=0.25, label="_nolegend_")
    plt.plot(rg_clean["calendar_year"], rg_trend.values,
             color="black", linewidth=3.0, label="Historical HP trend (2000-2025)")
    for label, df_sim in filtered.items():
        df_a = _add_anchor_and_calendar(df_sim)
        g = _band_by_year(df_a, "r_minus_g", time_col="calendar_year")
        plt.plot(g["time"], g["median"], label=label)
        plt.fill_between(g["time"], g["p25"], g["p75"], alpha=0.20)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.9)
    plt.xlim(debt_hist["calendar_year"].min(), SIM_START + n_years - 1)
    plt.xlabel("", fontsize=20); plt.ylabel("Interest-Growth Differential (r - g)", fontsize=20)
    plt.grid(True)
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout()
    plt.savefig(output / f"sdsa_enrichment1_rg_overlay_ai_booms_{c_val:.2f}.pdf", dpi=300)
    plt.close()

    # Snowball overlay
    plt.figure(figsize=(11,7))
    sb = snowball_hist.dropna(subset=["snowball"])
    _, sb_trend = sm.tsa.filters.hpfilter(sb["snowball"], lamb=100)
    plt.plot(sb["calendar_year"], sb["snowball"],
             color="black", linewidth=1.2, alpha=0.25, label="_nolegend_")
    plt.plot(sb["calendar_year"], sb_trend.values,
             color="black", linewidth=3.0, label="Historical HP trend (2000-2025)")
    for label, df_sim in filtered.items():
        df_a = _add_anchor_and_calendar(df_sim)
        g = _band_by_year(df_a, "snowball", time_col="calendar_year")
        plt.plot(g["time"], g["median"], label=label)
        plt.fill_between(g["time"], g["p25"], g["p75"], alpha=0.20)
    plt.axvline(2025, color="black", linestyle="--", linewidth=0.9, alpha=0.8,
                label="_nolegend_")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.9)
    plt.xlim(debt_hist["calendar_year"].min(), SIM_START + n_years - 1)
    plt.xlabel("", fontsize=20); plt.ylabel("Snowball Term", fontsize=20)
    plt.grid(True)
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout()
    plt.savefig(output / f"sdsa_enrichment1_snowball_overlay_ai_booms_{c_val:.2f}.pdf", dpi=300)
    plt.close()

# ---- Optional overlay for growth ----
plt.figure(figsize=(11,7))
for ai_label, params in ai_scenarios.items():
    plt.plot(years, params["a_ug"], label=ai_label)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.9)
plt.title("", fontsize=24)
plt.xlabel("", fontsize=20); plt.ylabel("a_ug (growth rate)", fontsize=20)
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(output / "sdsa_enrichment1_a_ug_paths_ai_booms.pdf", dpi=300)
plt.close()

_ai_compare_c0_lookup = [
    ("Baseline growth",   "CBO Baseline - Irresponsible (c=0.00)"),
    ("AI boom (+0.5pp)",  "AI Boom (0.5pp Productivity) - Irresponsible (c=0.00)"),
    ("AI boom (+1.0pp)",  "AI Boom (1.0pp Productivity) - Irresponsible (c=0.00)"),
]
ai_compare_c0 = {
    pretty: _add_anchor_and_calendar(sim_results_by_ai[key])
    for pretty, key in _ai_compare_c0_lookup
    if key in sim_results_by_ai
}

plot_primary_balance_story_popouts(
    scenarios=ai_compare_c0,
    s_hist=s_hist,
    output_path=output,
    sim_start_year=SIM_START,
    b0=b0,
    recession_df=recession_df,
    hist_start_year=1984,
    popout_xmax=2037,
    annotation_fontsize=18,
    fname="sdsa_primary_balance_story_popouts_ai_c_0_00.pdf",
    focus_year="end",
    popout_years=6,
    print_hist_improvement_stats=True,
)

# ============================================================
# Distribution of 10-Year Debt Changes — AI vs Baseline (c = 0.00 only)
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
        change = (df_sim["b"].iloc[-1] - b0) * 100
        changes.append(change)
    return np.array(changes)

# ── collect three scenarios (c = 0.00 only) ──────────────────
target_labels = [
    "CBO Baseline - Irresponsible (c=0.00)",
    "AI Boom (0.5pp Productivity) - Irresponsible (c=0.00)",
    "AI Boom (1.0pp Productivity) - Irresponsible (c=0.00)",
]
distros = {}
for label in target_labels:
    if label not in sim_results_by_ai:
        print(f"Warning: {label} not found in results")
        continue
    distros[label] = compute_ten_year_changes(sim_results_by_ai[label])

# ============================================================
# Plot: Distribution of 10-Year Debt/GDP Changes (pp) — AI vs Baseline (c = 0.00)
# ============================================================

plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=["#1f77b4", "#9467bd", "#8c564b", "#7f7f7f", "#17becf", "#ff7f0e"]
)
plt.figure(figsize=(12, 8))
graphing_labels = {
    "CBO Baseline - Irresponsible (c=0.00)": "Baseline Growth",
    "AI Boom (0.5pp Productivity) - Irresponsible (c=0.00)": "AI Boom (0.5pp Productivity)",
    "AI Boom (1.0pp Productivity) - Irresponsible (c=0.00)": "AI Boom (1.0pp Productivity)",
}
# simulated distributions (already computed in distros)
for lbl in target_labels:
    if lbl not in distros:
        continue
    sns.histplot(
        distros[lbl],
        label=graphing_labels.get(lbl, lbl),
        stat="density",
        element="step",
        fill=False,
        linewidth=1.5,
        alpha=1
    )
plt.axvline(x=0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
# historical distribution (bold black)
plt.title("", fontsize=24)
plt.xlabel("10-year change in Debt/GDP (percentage points)", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(loc="best", fontsize=18)
plt.tight_layout()
plt.savefig(output / "sdsa_enrichment1_debt_10yr_change_distro_ai_booms_c_0.00.pdf", dpi=300)
plt.show()