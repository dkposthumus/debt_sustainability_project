"""
robustness_checks.py
--------------------
Purpose : Robustness checks for the debt-sustainability paper. Each block
          substitutes an alternative assumption for one of the model's inputs
          and reports how the headline results move.
Inputs  : data/clean/master_projections_cleaned.csv         (CBO Feb 2026, clean)
          data/raw/cbo_budget_projections.xlsx              (CBO net interest,
                                                             federal revenues/outlays)
          data/raw/imf_us_article_iv_apr2026.csv            (IMF Article IV
                                                             federal-level table)
          data/raw/WEOApr2026all.xlsx                       (IMF Apr 2026 WEO,
                                                             general government)
          data/raw/WEOOct2025all.csv                        (IMF Oct 2025 WEO,
                                                             general government)
Outputs : output/exploratory/robustness_federal_cbo_vs_imf.{pdf,csv}
              CBO Feb 2026 federal baseline vs. IMF Apr 2026 Article IV federal —
              apples-to-apples federal scope, 3 panels (primary balance, real
              10-yr Treasury, real GDP growth).
          output/exploratory/robustness_fiscal_cbo_vs_imf.{pdf,csv}
              CBO Feb 2026 federal baseline vs. IMF Apr 2026 WEO general gov.
              vs. IMF Oct 2025 WEO general gov. — 3 panels (primary balance,
              revenues, primary outlays). The two IMF lines are gen-gov scope;
              CBO is federal. Useful when a reader (e.g., Blanchard's comment)
              quotes an IMF WEO primary deficit "near 4%" — that's the gen-gov
              number, ~1.3 pp wider than the federal number by scope alone.
Author  : Dan Post
Created : 2026-05-12

Notes
-----
Sign convention: primary balance = signed primary surplus (positive = surplus).

For the federal chart we derive IMF's federal **primary** balance from the
published federal **total** fiscal balance by adding back CBO's projected
federal net interest (proxy). IMF doesn't break out federal interest, and
federal interest is largely a function of debt level and rates, both of which
IMF and CBO project very similarly — the proxy adds <0.3 pp of noise.

For the gen-gov chart, primary outlays are derived as revenues − primary
balance for both IMF vintages (algebraic identity).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

work_dir    = Path(__file__).resolve().parent.parent
raw_data    = work_dir / 'data' / 'raw'
clean_data  = work_dir / 'data' / 'clean'
exploratory = work_dir / 'output' / 'exploratory'
exploratory.mkdir(parents=True, exist_ok=True)

plt.style.use(str(Path(__file__).resolve().parent / 'mahoney_lab.mplstyle'))

CBO_CLEAN_PATH       = clean_data / 'master_projections_cleaned.csv'
CBO_BUDGET_PATH      = raw_data / 'cbo_budget_projections.xlsx'
IMF_ART_IV_PATH      = raw_data / 'imf_us_article_iv_apr2026.csv'
IMF_WEO_APR2026_PATH = raw_data / 'WEOApr2026all.xlsx'
IMF_WEO_OCT2025_PATH = raw_data / 'WEOOct2025all.csv'

################################################################################
## CBO loaders
################################################################################

def load_cbo_clean() -> pd.DataFrame:
    """Quarterly master projections from `01_gather_projections.py` output."""
    df = pd.read_csv(CBO_CLEAN_PATH, parse_dates=['date'])
    df['cal_year'] = df['date'].dt.year
    return df


def cbo_primary_balance_annual() -> pd.DataFrame:
    """CBO Feb 2026 federal primary balance (% of GDP), annual."""
    df = load_cbo_clean()
    pb = (df.dropna(subset=['year', 's (cbo baseline)'])
            .loc[:, ['year', 's (cbo baseline)']]
            .rename(columns={'s (cbo baseline)': 'cbo_primary_balance'}))
    pb['year'] = pb['year'].astype(int)
    return pb.drop_duplicates('year').reset_index(drop=True)


def cbo_real_10yr_annual() -> pd.DataFrame:
    """Annual avg of CBO Feb 2026 quarterly real 10-yr Treasury rate
    (10-yr nominal − GDP price index inflation)."""
    df = load_cbo_clean()
    return (df.dropna(subset=['r (cbo baseline)'])
              .groupby('cal_year', as_index=False)['r (cbo baseline)']
              .mean()
              .rename(columns={'cal_year': 'year',
                               'r (cbo baseline)': 'cbo_real_10yr'}))


def cbo_real_growth_annual() -> pd.DataFrame:
    """Annual avg of CBO Feb 2026 quarterly real GDP growth (annualized)."""
    df = load_cbo_clean()
    return (df.dropna(subset=['g (cbo baseline)'])
              .groupby('cal_year', as_index=False)['g (cbo baseline)']
              .mean()
              .rename(columns={'cal_year': 'year',
                               'g (cbo baseline)': 'cbo_real_growth'}))


def _cbo_table_1_1_pct_gdp(raw: pd.DataFrame) -> dict:
    """Helper: locate row indices in CBO Table 1-1 % of GDP block."""
    def _has_years(row):
        nums = row.dropna().apply(lambda v: str(v).replace('.0', ''))
        return nums.str.match(r'^\d{4}$').sum() >= 5
    year_row_idx = raw.apply(_has_years, axis=1).idxmax()
    headers = raw.iloc[year_row_idx].tolist()
    year_cols = [(i, int(h)) for i, h in enumerate(headers)
                 if isinstance(h, (int, float)) and 2020 <= h <= 2050]

    def _label(i):
        for col in (0, 1):
            v = raw.iat[i, col]
            if not pd.isna(v):
                return str(v).strip()
        return ''
    pct_start = next((i for i in range(len(raw))
                      if _label(i).lower() == 'as a percentage of gdp'), None)
    if pct_start is None:
        raise ValueError("Could not find % of GDP section in Table 1-1.")
    return {'year_cols': year_cols, 'pct_start': pct_start, 'label_fn': _label}


def cbo_federal_fiscal_pct_gdp() -> pd.DataFrame:
    """CBO Feb 2026 federal revenues, total outlays, net interest, and primary
    outlays as % of GDP. From Table 1-1 % of GDP block (unadjusted)."""
    raw = pd.read_excel(CBO_BUDGET_PATH, sheet_name='Table 1-1', header=None,
                        engine='openpyxl')
    meta = _cbo_table_1_1_pct_gdp(raw)
    year_cols = meta['year_cols']
    pct_start = meta['pct_start']
    _label = meta['label_fn']

    def _find_row(label_pattern: str, after: int) -> int:
        pat = re.compile(label_pattern, re.IGNORECASE)
        for i in range(after, len(raw)):
            if pat.search(_label(i)):
                return i
        raise ValueError(f"Row not found after {after}: {label_pattern!r}")

    revenues_hdr = _find_row(r'^\s*Revenues\s*$', pct_start)
    revenues_tot = _find_row(r'^\s*Total\s*$',  revenues_hdr)
    outlays_hdr  = _find_row(r'^\s*Outlays\s*$', revenues_tot)
    net_interest = _find_row(r'^\s*Net interest\s*$', outlays_hdr)
    outlays_tot  = _find_row(r'^\s*Total\s*$',  net_interest)

    def _row_vals(i): return [float(raw.iat[i, c]) for c, _ in year_cols]

    df = pd.DataFrame({
        'year':              [yr for _, yr in year_cols],
        'cbo_revenues':      _row_vals(revenues_tot),
        'cbo_outlays_total': _row_vals(outlays_tot),
        'cbo_net_interest':  _row_vals(net_interest),
    })
    df['cbo_primary_outlays'] = df['cbo_outlays_total'] - df['cbo_net_interest']
    df['cbo_primary_balance_unadj'] = df['cbo_revenues'] - df['cbo_primary_outlays']
    return df


################################################################################
## IMF Article IV loader and derivations
################################################################################

def load_imf_article_iv_wide() -> pd.DataFrame:
    """Returns the IMF Article IV table reshaped wide: year × indicator."""
    df = pd.read_csv(IMF_ART_IV_PATH, comment='#')
    long = df.melt(id_vars='indicator', var_name='year', value_name='value')
    long['year'] = long['year'].astype(int)
    long['value'] = pd.to_numeric(long['value'], errors='coerce')
    return long.pivot(index='year', columns='indicator', values='value').reset_index()


def imf_federal_primary_balance() -> pd.DataFrame:
    """IMF federal primary balance ≈ IMF federal fiscal balance + CBO federal
    net interest (proxy)."""
    art = load_imf_article_iv_wide()[['year', 'federal_fiscal_balance_pct_gdp']]
    interest = cbo_federal_fiscal_pct_gdp()[['year', 'cbo_net_interest']]
    out = art.merge(interest, on='year', how='left')
    out['imf_federal_primary_balance'] = (
        out['federal_fiscal_balance_pct_gdp'] + out['cbo_net_interest']
    )
    return out[['year', 'imf_federal_primary_balance']]


def imf_real_10yr_and_growth() -> pd.DataFrame:
    """IMF real 10-yr = nominal 10-yr (Q4 avg) − PCE Q4/Q4 inflation.
       IMF real growth = published annual real GDP growth."""
    art = load_imf_article_iv_wide()
    art['imf_real_10yr'] = (art['ten_year_govt_bond_rate_q4_avg']
                            - art['pce_inflation_q4q4'])
    return art[['year', 'imf_real_10yr', 'real_gdp_growth_annual']] \
            .rename(columns={'real_gdp_growth_annual': 'imf_real_growth'})


################################################################################
## IMF WEO loaders (general-government scope)
################################################################################

def load_imf_weo_apr2026_us() -> pd.DataFrame:
    """US general-gov revenues, expenditure, primary balance from IMF Apr 2026
    WEO (`WEOApr2026all.xlsx`). Returns year × {gg_revenues, gg_outlays_total,
    gg_primary_balance, gg_primary_outlays}, all % of GDP."""
    raw = pd.read_excel(IMF_WEO_APR2026_PATH, sheet_name='Countries', header=0,
                        engine='openpyxl')
    year_cols = [c for c in raw.columns
                 if isinstance(c, (int, float)) and 1980 <= c <= 2050]
    us = raw[raw['COUNTRY.ID'] == 'USA']

    def _series(code: str) -> dict:
        row = us[us['INDICATOR.ID'] == code].iloc[0]
        return {int(y): pd.to_numeric(row[y], errors='coerce') for y in year_cols}

    rev = _series('GGR_NGDP')
    out = _series('GGX_NGDP')
    pb  = _series('GGXONLB_NGDP')
    years = sorted(rev.keys())
    df = pd.DataFrame({
        'year':               years,
        'imf_apr2026_gg_revenues':        [rev[y] for y in years],
        'imf_apr2026_gg_outlays_total':   [out[y] for y in years],
        'imf_apr2026_gg_primary_balance': [pb[y]  for y in years],
    })
    df['imf_apr2026_gg_primary_outlays'] = (df['imf_apr2026_gg_revenues']
                                             - df['imf_apr2026_gg_primary_balance'])
    return df


def load_imf_weo_oct2025_us() -> pd.DataFrame:
    """US general-gov revenues, expenditure, primary balance, and real GDP
    growth from IMF Oct 2025 WEO (`WEOOct2025all.csv`)."""
    df = pd.read_csv(IMF_WEO_OCT2025_PATH, low_memory=False)
    year_cols = [str(y) for y in range(1980, 2032) if str(y) in df.columns]
    us = df[df['COUNTRY'].astype(str).str.contains('United States', na=False)]

    def _series(series_code: str) -> dict:
        sub = us[us['SERIES_CODE'] == series_code]
        if sub.empty:
            raise ValueError(f"Series {series_code} not found for US in Oct 2025 WEO.")
        row = sub.iloc[0]
        return {int(y): pd.to_numeric(row[y], errors='coerce') for y in year_cols}

    rev = _series('USA.GGR_NGDP.A')
    out = _series('USA.GGX_NGDP.A')
    pb  = _series('USA.GGXONLB_NGDP.A')
    growth = _series('USA.NGDP_RPCH.A')
    years = sorted(rev.keys())
    df_out = pd.DataFrame({
        'year':                           years,
        'imf_oct2025_gg_revenues':        [rev[y]    for y in years],
        'imf_oct2025_gg_outlays_total':   [out[y]    for y in years],
        'imf_oct2025_gg_primary_balance': [pb[y]     for y in years],
        'imf_oct2025_real_growth':        [growth[y] for y in years],
    })
    df_out['imf_oct2025_gg_primary_outlays'] = (df_out['imf_oct2025_gg_revenues']
                                                  - df_out['imf_oct2025_gg_primary_balance'])
    return df_out


################################################################################
## Comparison + plotting
################################################################################

def build_federal_comparison() -> pd.DataFrame:
    """Federal-scope: CBO Feb 2026 baseline vs. IMF Apr 2026 Article IV."""
    cbo = (cbo_primary_balance_annual()
           .merge(cbo_real_10yr_annual(),   on='year', how='outer')
           .merge(cbo_real_growth_annual(), on='year', how='outer'))
    imf = (imf_federal_primary_balance()
           .merge(imf_real_10yr_and_growth(), on='year', how='outer'))
    return cbo.merge(imf, on='year', how='outer').sort_values('year').reset_index(drop=True)


def build_fiscal_comparison() -> pd.DataFrame:
    """Gen-gov-scope (for IMF series) vs. CBO federal: primary balance, revenues,
    primary outlays. CBO is federal-only; IMF series are general government."""
    cbo = cbo_federal_fiscal_pct_gdp()[['year', 'cbo_revenues',
                                         'cbo_primary_outlays',
                                         'cbo_primary_balance_unadj']]
    apr = load_imf_weo_apr2026_us()[['year', 'imf_apr2026_gg_revenues',
                                      'imf_apr2026_gg_primary_outlays',
                                      'imf_apr2026_gg_primary_balance']]
    oct_ = load_imf_weo_oct2025_us()[['year', 'imf_oct2025_gg_revenues',
                                        'imf_oct2025_gg_primary_outlays',
                                        'imf_oct2025_gg_primary_balance']]
    return (cbo.merge(apr, on='year', how='outer')
               .merge(oct_, on='year', how='outer')
               .sort_values('year').reset_index(drop=True))


def plot_federal_comparison(df: pd.DataFrame,
                            year_min: int = 2024, year_max: int = 2031) -> Path:
    d = df[(df['year'] >= year_min) & (df['year'] <= year_max)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True)
    panels = [
        ('cbo_primary_balance', 'imf_federal_primary_balance',
         'Primary balance (% of GDP)', 'Federal primary balance'),
        ('cbo_real_10yr',       'imf_real_10yr',
         'Real 10-yr Treasury rate (%)', 'Real 10-year rate'),
        ('cbo_real_growth',     'imf_real_growth',
         'Real GDP growth (%)', 'Real GDP growth'),
    ]
    for ax, (cbo_col, imf_col, ylabel, title) in zip(axes, panels):
        m_cbo = d[cbo_col].notna()
        ax.plot(d.loc[m_cbo, 'year'], d.loc[m_cbo, cbo_col],
                marker='o', linewidth=2.2, label='CBO Feb 2026 baseline')
        m_imf = d[imf_col].notna()
        ax.plot(d.loc[m_imf, 'year'], d.loc[m_imf, imf_col],
                marker='s', linewidth=2.2, linestyle='--',
                label='IMF Apr 2026 Article IV')
        if 'balance' in cbo_col:
            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.6)
        ax.set_title(title); ax.set_xlabel('Year'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc='best', frameon=False, fontsize=11)
    fig.suptitle('CBO vs. IMF Article IV — federal projections (robustness)',
                 y=1.02)
    fig.tight_layout()
    out_path = exploratory / 'robustness_federal_cbo_vs_imf.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_fiscal_comparison(df: pd.DataFrame,
                           year_min: int = 2024, year_max: int = 2031) -> Path:
    d = df[(df['year'] >= year_min) & (df['year'] <= year_max)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True)
    # (cbo_col, apr_col, oct_col, ylabel, title)
    panels = [
        ('cbo_primary_balance_unadj',
         'imf_apr2026_gg_primary_balance',
         'imf_oct2025_gg_primary_balance',
         'Primary balance (% of GDP)', 'Primary balance'),
        ('cbo_revenues',
         'imf_apr2026_gg_revenues',
         'imf_oct2025_gg_revenues',
         'Revenues (% of GDP)', 'Revenues'),
        ('cbo_primary_outlays',
         'imf_apr2026_gg_primary_outlays',
         'imf_oct2025_gg_primary_outlays',
         'Primary outlays (% of GDP)', 'Primary outlays'),
    ]
    for ax, (cbo_col, apr_col, oct_col, ylabel, title) in zip(axes, panels):
        m_cbo = d[cbo_col].notna()
        ax.plot(d.loc[m_cbo, 'year'], d.loc[m_cbo, cbo_col],
                marker='o', linewidth=2.2, label='CBO Feb 2026 (federal)')
        m_apr = d[apr_col].notna()
        ax.plot(d.loc[m_apr, 'year'], d.loc[m_apr, apr_col],
                marker='s', linewidth=2.2, linestyle='--',
                label='IMF Apr 2026 WEO (general gov.)')
        m_oct = d[oct_col].notna()
        ax.plot(d.loc[m_oct, 'year'], d.loc[m_oct, oct_col],
                marker='^', linewidth=2.2, linestyle=':',
                label='IMF Oct 2025 WEO (general gov.)')
        if 'balance' in cbo_col:
            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.6)
        ax.set_title(title); ax.set_xlabel('Year'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc='best', frameon=False, fontsize=10)
    fig.suptitle('CBO (federal) vs. IMF WEO (general gov., two vintages)',
                 y=1.02)
    fig.tight_layout()
    out_path = exploratory / 'robustness_fiscal_cbo_vs_imf.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    # --- Federal-scope chart (apples-to-apples) ---
    fed_df = build_federal_comparison()
    fed_csv = exploratory / 'robustness_federal_cbo_vs_imf.csv'
    fed_df.to_csv(fed_csv, index=False)
    fed_pdf = plot_federal_comparison(fed_df)
    print(f"Wrote {fed_csv.relative_to(work_dir)}")
    print(f"Wrote {fed_pdf.relative_to(work_dir)}")

    # --- Gen-gov-scope chart (CBO federal vs. IMF Apr 2026 + Oct 2025 WEO) ---
    fis_df = build_fiscal_comparison()
    fis_csv = exploratory / 'robustness_fiscal_cbo_vs_imf.csv'
    fis_df.to_csv(fis_csv, index=False)
    fis_pdf = plot_fiscal_comparison(fis_df)
    print(f"Wrote {fis_csv.relative_to(work_dir)}")
    print(f"Wrote {fis_pdf.relative_to(work_dir)}")

    overlap = fis_df[(fis_df['year'] >= 2024) & (fis_df['year'] <= 2031)]
    print("\nFiscal chart — overlap window (2024–2031, % of GDP):")
    print(overlap[['year',
                   'cbo_primary_balance_unadj',
                   'imf_apr2026_gg_primary_balance',
                   'imf_oct2025_gg_primary_balance']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
