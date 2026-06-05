"""
01b_gather_projections_imf.py
-----------------------------
Purpose : Build the IMF-Article-IV mean-path inputs (a_g, a_r, a_s) for the SDSA
          Monte Carlo, mirroring the column layout of
          `master_projections_cleaned.csv` so that `03_run_simulations_imf.py`
          can consume them with minimal changes.

          This is the IMF analogue of `01_gather_projections.py`. The ONLY thing
          that changes between the CBO and IMF runs is the exogenous mean path of
          the three core inputs; every model parameter is held fixed.

Inputs  : data/raw/imf_us_article_iv_apr2026.csv   (IMF Apr 2026 Article IV, 2024-2031)
          data/raw/cbo_budget_projections.xlsx      (CBO federal net interest, used
                                                     ONLY as a proxy to net out
                                                     interest from IMF's OVERALL
                                                     federal balance -> PRIMARY)
Output  : data/clean/master_projections_imf_cleaned.csv
              annual columns (PROJ window 2026-2031):
                year,
                g (imf baseline)   real GDP growth, % (a_g)
                r (imf baseline)   real 10-yr rate, % (a_r)
                s (imf baseline)   primary balance, % of GDP, SURPLUS POSITIVE (a_s)
                source             'imf' (2026-2031)

Conversion / design decisions (see also report and robustness_checks.py):
  1. REAL interest rate: IMF gives the NOMINAL 10-yr govt bond rate (Q4 avg). The
     model needs a REAL rate. We deflate by IMF's PCE Q4/Q4 inflation:
         r_real = nominal_10yr - pce_inflation_q4q4
     This matches `imf_real_10yr_and_growth()` in robustness_checks.py exactly.
     (CBO's run deflates by the GDP price index; IMF does not publish a GDP
     deflator path in this table, so PCE is the consistent IMF-internal choice.)

  2. PRIMARY balance: IMF gives the OVERALL federal fiscal balance (deficit
     negative, INCLUDES net interest). The model needs the PRIMARY balance
     (excludes net interest), surplus positive. We derive
         primary = overall_balance + net_interest
     IMF does not break out federal net interest, so we reuse CBO's projected
     federal net interest as a proxy — the SAME convention used in
     robustness_checks.py (`imf_federal_primary_balance`). Federal net interest
     is largely a function of the debt level and rates, which IMF and CBO project
     very similarly, so the proxy adds <~0.3 pp of noise. We use FEDERAL
     (debt-held-by-public) measures throughout to stay consistent with the
     model's b — NOT general-government.

  3. HORIZON: IMF Article IV data ends in 2031. The IMF run STOPS at that
     endpoint — we do NOT flat-extend (or otherwise impute) values for 2032-2036.
     The projection simply ends where the source projections end, so the IMF run
     covers a 6-year horizon (2026-2031) versus the CBO run's 10-year horizon
     (2026-2036). Figures and AA are reported through 2031 accordingly.
"""

import importlib.util
import pandas as pd
from pathlib import Path

work_dir   = Path(__file__).resolve().parent.parent
raw_data   = work_dir / 'data' / 'raw'
clean_data = work_dir / 'data' / 'clean'
clean_data.mkdir(parents=True, exist_ok=True)

PROJ_START = 2026
IMF_LAST   = 2031   # last year IMF Article IV table covers
PROJ_END   = IMF_LAST   # IMF run stops at the IMF endpoint — no flat extension to 2036

IMF_ART_IV_PATH = raw_data / 'imf_us_article_iv_apr2026.csv'

# ---------------------------------------------------------------------------
# Reuse robustness_checks.py's vetted IMF/CBO derivations so the conventions
# are byte-for-byte identical to the team's existing comparison logic.
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    'robustness_checks', Path(__file__).resolve().parent / 'robustness_checks.py')
_rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rc)

# ---------------------------------------------------------------------------
# 1-2. Real rate, real growth, and derived primary balance (2024-2031)
# ---------------------------------------------------------------------------
rate_growth = _rc.imf_real_10yr_and_growth()           # year, imf_real_10yr, imf_real_growth
primary     = _rc.imf_federal_primary_balance()        # year, imf_federal_primary_balance
                                                       # (overall bal + CBO net interest)

imf = (rate_growth.merge(primary, on='year', how='outer')
       .sort_values('year').reset_index(drop=True))

imf = imf.rename(columns={
    'imf_real_growth':              'g (imf baseline)',
    'imf_real_10yr':                'r (imf baseline)',
    'imf_federal_primary_balance':  's (imf baseline)',
})

# Restrict to the projection window the IMF Article IV table covers (2026-2031).
# The IMF run STOPS at the IMF endpoint; we do NOT flat-extend to 2036, so the
# projection simply ends where the source projections end.
imf_proj = imf[(imf['year'] >= PROJ_START) & (imf['year'] <= IMF_LAST)].copy()
imf_proj['source'] = 'imf'

master_imf = (imf_proj[['year', 'g (imf baseline)', 'r (imf baseline)',
                        's (imf baseline)', 'source']]
              .sort_values('year').reset_index(drop=True))

out_path = clean_data / 'master_projections_imf_cleaned.csv'
master_imf.to_csv(out_path, index=False)

print(f"Wrote {out_path.relative_to(work_dir)} ({len(master_imf)} rows, "
      f"{PROJ_START}-{PROJ_END}).")
print(master_imf.round(3).to_string(index=False))
