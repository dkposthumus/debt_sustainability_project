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

debt_hist = get_fred_series('FYGFGDQ188S', 'debt_pct_gdp')
debt_hist = debt_hist[debt_hist['date'] >= '2005-01-01'].copy()
# FYGFGDQ188S is percent of GDP; convert to ratio to match model b
debt_hist['b_hist'] = debt_hist['debt_pct_gdp'] / 100.0
# Use Q4 (October) as your "annual" observation, consistent with your 10y change code
debt_hist = debt_hist[debt_hist['date'].dt.month == 10].copy()
debt_hist['calendar_year'] = debt_hist['date'].dt.year
# Keep 2005–2025 (inclusive)
debt_hist = debt_hist[(debt_hist['calendar_year'] >= 2005) &
                      (debt_hist['calendar_year'] <= 2025)][['calendar_year', 'b_hist']].dropna()

growth = get_fred_series('A191RL1Q225SBEA', 'gdp_growth_rate') # quarterly, percent (real)
interest = get_fred_series('REAINTRATREARAT10Y', 'interest_rate') # monthly, percent (real)
snowball_hist = growth.merge(interest, on='date', how='outer')
# convert to calendar year
snowball_hist = snowball_hist[snowball_hist['date'].dt.month == 10].copy()
snowball_hist['calendar_year'] = snowball_hist['date'].dt.year
snowball_hist = snowball_hist[(snowball_hist['calendar_year'] >= 2005) &
                                (snowball_hist['calendar_year'] <= 2025)].copy()
snowball_hist = debt_hist.merge(snowball_hist, on='calendar_year', how='left')
snowball_hist['snowball'] = (snowball_hist['interest_rate'] - snowball_hist['gdp_growth_rate']) / 100.0 * snowball_hist['b_hist']

# -------------------------------------------------
# Core simulator (FIRST-DIFFERENCE r law)
# -------------------------------------------------
def simulate_scenario(
    c_val,
    a_s_vec, a_ug,
    r_star, beta_r, rho, sigma,
    s_g, s_x, s_r, s_s,
    x0=0.0, r0=None, b0=0.9656,
    n_years=10, n_simulations=20000, label=""
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
np.random.seed(42)

# horizons and baseline levels
n_years = len(a_ug)
n_sims  = 20000
b0      = 0.9656
r_star  = 0.01
r0      = 0.02

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
                       r_star: float, b0: float,
                       debt_hist: pd.DataFrame,
                       snowball_hist: pd.DataFrame,
                       sim_start_year: int = 2025):

    graphics_path.mkdir(parents=True, exist_ok=True)

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
        for var in ylim_store:
            ylim_store[var].extend(df_sim[var].values)

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
                color="black", linewidth=3.0, label="Historical (FRED, 2005–2025)"
            )
            plt.axvline(sim_start_year, color="black", linestyle="--", linewidth=0.9, alpha=0.8,
                    label="_nolegend_")
        if var == "snowball":
            plt.plot(
                snowball_hist["calendar_year"], snowball_hist["snowball"],
                color="black", linewidth=3.0, label="Historical (2005–2025)"
            )
            plt.axvline(sim_start_year, color="black", linestyle="--", linewidth=0.9, alpha=0.8,
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
    plt.title('')
    plt.xlabel('')
    plt.ylabel('real growth rate (g)')
    plt.grid(True)
    plt.legend(loc='best', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(graphics_path / 'sdsa_enrichment1_g_rates_responsible_only.pdf', dpi=300)
    plt.close()

    '''_plot_overlay('r',   # uses r_av internally
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
                  fname='sdsa_enrichment1_rg_overlay.pdf')'''

    _plot_overlay('b',
                  'Debt (b): Median & IQR — Regimes',
                  'b',
                  yline=b0,
                  ylim_key='b',
                  fname='sdsa_enrichment1_b_overlay.pdf')

    '''_plot_overlay('interest_share',
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
                  fname='sdsa_enrichment1_curvature_overlay.pdf')'''
    
    _plot_overlay(
        'snowball',
        'Snowball term: (r - g) × b',
        'Snowball term',
        yline=0.0,
        ylim_key='snowball',
        fname='sdsa_enrichment1_snowball_overlay.pdf'
    )

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
enriched_frames = summarize_and_plot(sim_results_by_regime, output, d_dict, r_star, b0,
                                     debt_hist=debt_hist, snowball_hist=snowball_hist,
                                     sim_start_year=2025)
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
    popout_xmax: int = 2036,
    shortened_recessions: bool = False,
    annotation_fontsize: int = 15,
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
      - Pop-out x-axis extends to popout_xmax (default 2036)
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

    for i, (sc_label, df_sim) in enumerate(scenarios.items()):
        df = df_sim.copy()
        if "calendar_year" not in df.columns:
            df["calendar_year"] = sim_start_year + df["year"] - 1

        df = add_stabilizing_and_adjustment(df, b0=b0)

        s_med = summarize_median(df, "s", time_col="calendar_year")
        st_med = summarize_median(df, "s_stab", time_col="calendar_year")

        # convert to pp + datetime x
        s_med["median"] *= 100.0
        st_med["median"] *= 100.0
        s_med["date"] = pd.to_datetime(s_med["calendar_year"].astype(int).astype(str) + "-01-01")
        st_med["date"] = pd.to_datetime(st_med["calendar_year"].astype(int).astype(str) + "-01-01")

        color = color_cycle[i % len(color_cycle)]

        ax.plot(s_med["date"], s_med["median"], linewidth=2.6, color=color, label="_nolegend_")
        ax.plot(st_med["date"], st_med["median"], linewidth=2.6, linestyle="--", color=color, label="_nolegend_")

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
    ax.set_xlabel("")
    ax.set_ylabel("Primary balance (% of GDP; surplus positive)")
    ax.set_title("")  # per your request
    ax.grid(True, alpha=0.35)

    # legend: ONLY scenario colors + historical entry
    handles = [Line2D([0], [0], color="black", linewidth=2.5, label="Historical primary balance")]
    for sc_label, (_, _, color) in scenario_summaries.items():
        handles.append(Line2D([0], [0], color=color, linewidth=3.0, label=sc_label))
    ax.legend(handles=handles, loc="best", fontsize="large")

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
        st_win = st_med[(st_med["calendar_year"] >= x0_year) & (st_med["calendar_year"] <= focus_year_val)].copy()

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

        ax_in.tick_params(labelsize=9)
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

recession_df = get_fred_series("USREC", "recession")  # date: datetime, recession: 0/1

c_regimes = {
    f"Irresponsible (c={d_dict['irresponsible']:.2f})": sim_results_by_regime["irresponsible"],
    f"Responsible (c={d_dict['responsible']:.2f})": sim_results_by_regime["responsible"],
    f"Very responsible (c={d_dict['very responsible']:.2f})": sim_results_by_regime["very responsible"],
}

plot_primary_balance_story_popouts(
    scenarios=c_regimes,
    s_hist=s_hist,
    output_path=output,
    sim_start_year=2025,
    b0=b0,
    recession_df=recession_df,
    hist_start_year=1984,
    focus_year="end",
    popout_xmax=2036,
    annotation_fontsize=15,
    fname="sdsa_primary_balance_story_popouts_c_regimes.pdf",
    shortened_recessions=False,
    print_hist_improvement_stats=True,
)

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
    "Responsible (c=0.15)": 0.15,
    "Very Responsible (c=0.30)": 0.30
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
SIM_START_YEAR = 2025  # matches your simulation start

for c_label, c_val in c_scenarios.items():
    filtered = {k: v for k, v in sim_results_by_ai.items() if c_label in k}
    # -------------------------
    # Debt overlay (historical + simulated)
    # -------------------------
    plt.figure(figsize=(11,7))
    # historical
    plt.plot(
        debt_hist["calendar_year"], debt_hist["b_hist"],
        color="black", linewidth=3.0, label="Historical (FRED, 2000–2025)"
    )
    plt.axvline(SIM_START_YEAR, color="black", linestyle="--", linewidth=0.9, alpha=0.8, label="_nolegend_")
    # simulated
    for label, df_sim in filtered.items():
        df_sim = df_sim.copy()
        df_sim["calendar_year"] = SIM_START_YEAR + df_sim["year"] - 1
        g = _band_by_year(df_sim, "b", time_col="calendar_year")  # << force calendar years
        plt.plot(g["time"], g["median"], label=label)
        plt.fill_between(g["time"], g["p25"], g["p75"], alpha=0.20)
    plt.axhline(y=b0, color="black", linestyle="--", linewidth=0.9)
    plt.xlim(debt_hist["calendar_year"].min(), SIM_START_YEAR + n_years - 1)  # ensures historical is in frame
    plt.ylim(b0 - 0.1, 1.7)
    plt.xlabel(""); plt.ylabel("Debt-to-GDP Ratio (b)")
    plt.grid(True)
    plt.legend(loc="best", fontsize="x-large")
    plt.tight_layout()
    plt.savefig(output / f"sdsa_enrichment1_b_overlay_ai_booms_{c_val:.2f}.pdf", dpi=300)
    plt.close()

    # -------------------------
    # Snowball overlay (historical + simulated)
    # -------------------------
    plt.figure(figsize=(11,7))
    # historical
    sb = snowball_hist.dropna(subset=["snowball"])
    plt.plot(
        sb["calendar_year"], sb["snowball"],
        color="black", linewidth=3.0, label="Historical (2000–2025)"
    )
    plt.axvline(SIM_START_YEAR, color="black", linestyle="--", linewidth=0.9, alpha=0.8, label="_nolegend_")
    # simulated
    for label, df_sim in filtered.items():
        df_sim = df_sim.copy()
        df_sim["calendar_year"] = SIM_START_YEAR + df_sim["year"] - 1
        df_sim["snowball"] = (df_sim["r_av"] - df_sim["g"]) * df_sim.groupby("sim")["b"].shift(1).fillna(b0)
        g = _band_by_year(df_sim, "snowball", time_col="calendar_year")
        plt.plot(g["time"], g["median"], label=label)
        plt.fill_between(g["time"], g["p25"], g["p75"], alpha=0.20)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.9)
    plt.xlim(debt_hist["calendar_year"].min(), SIM_START_YEAR + n_years - 1)
    plt.xlabel(""); plt.ylabel("Snowball Term")
    plt.grid(True)
    plt.legend(loc="best", fontsize="x-large")
    plt.tight_layout()
    plt.savefig(output / f"sdsa_enrichment1_snowball_overlay_ai_booms_{c_val:.2f}.pdf", dpi=300)
    plt.show()

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

ai_compare_c0 = {
    "Baseline growth": sim_results_by_ai["CBO Baseline - Irresponsible (c=0.00)"],
    "AI boom (+0.5pp)": sim_results_by_ai["AI Boom (0.5pp Productivity) - Irresponsible (c=0.00)"],
    "AI boom (+1.0pp)": sim_results_by_ai["AI Boom (1.0pp Productivity) - Irresponsible (c=0.00)"],
}

plot_primary_balance_story_popouts(
    scenarios=ai_compare_c0,
    s_hist=s_hist,
    output_path=output,
    sim_start_year=2025,
    b0=b0,
    recession_df=recession_df,
    hist_start_year=1984,
    popout_xmax=2036,
    annotation_fontsize=15,
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
        change = (df_sim["b"].iloc[-1] - df_sim["b"].iloc[0]) * 100
        changes.append(change)
    return np.array(changes)

# ── collect three scenarios (c = 0.15 only) ──────────────────
target_labels = [
    "CBO Baseline - Irresponsible (c=0.00)",
    "AI Boom (0.5pp Productivity) - Irresponsible (c=0.00)",
    "CBO Baseline - Responsible (c=0.15)",
    "AI Boom (1.0pp Productivity) - Very Responsible (c=0.30)",
    "AI Boom (1.0pp Productivity) - Irresponsible (c=0.00)",
    "CBO Baseline - Very Responsible (c=0.30)"
]
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
# historical distribution (bold black)
sns.histplot(
    historical_changes,
    label="Historical",
    stat="density",
    element="step",
    fill=False,
    linewidth=3.0,
    alpha=1,
    color="black"
)
plt.title("")
plt.xlabel("10-year change in Debt/GDP (percentage points)")
plt.ylabel("Density")
plt.legend(loc="best", fontsize="x-large")
plt.tight_layout()
plt.savefig(output / "sdsa_enrichment1_debt_10yr_change_distro_ai_booms_c_0.00.pdf", dpi=300)
plt.show()