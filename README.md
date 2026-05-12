# A Stochastic Framework for U.S. Debt Sustainability — Replication Package

Replication code and data for Bernstein, Posthumus, and Shaw, *A Stochastic Framework for U.S. Debt Sustainability*.

## Repository layout

```
.
├── code/
│   ├── 01_gather_projections.py     # builds master_projections_cleaned.csv
│   ├── 02_estimate_term_premium.py  # term premium regression -> β_r, ρ
│   ├── 03_run_simulations.py        # Monte Carlo, produces all paper figures
│   └── mahoney_lab.mplstyle
├── data/
│   ├── raw/        # all hand-curated / externally downloaded inputs
│   └── clean/      # outputs of 01_gather_projections.py
├── output/
│   └── figures/    # paper figures (produced by 03_run_simulations.py)
└── debt_sustainability_whitepaper_dp.tex
```

## Reproducing the figures

```bash
pip install -r requirements.txt
export FRED_API_KEY=<your-key>     # https://fred.stlouisfed.org/docs/api/api_key.html
python code/01_gather_projections.py
python code/02_estimate_term_premium.py
python code/03_run_simulations.py
```

Run in order. Scripts derive their paths from the script file's location, so the invocation directory does not matter.

## Data sources (`data/raw/`)

| File | Vintage | Source |
|---|---|---|
| `cbo_econ_projections.xlsx` | Feb 2026 | CBO, *The Budget and Economic Outlook: 2026 to 2036*, [economic projections](https://www.cbo.gov/publication/61882) |
| `cbo_budget_projections.xlsx` | Feb 2026 | CBO, *The Budget and Economic Outlook: 2026 to 2036*, [budget projections](https://www.cbo.gov/publication/61882) |
| `cbo_ai_projections.xlsx` | May 2025 | CBO, [*The Long-Term Budget Outlook Under Alternative Scenarios for the Economy and the Budget*](https://www.cbo.gov/publication/61332). Reference only — used to build `ai_scenarios.csv`; not read by the pipeline directly. |
| `ai_scenarios.csv` | — | AI productivity-boom growth and primary-deficit-delta paths (0.5pp and 1.0pp) for 2026–2035. **Source / derivation needs to be documented** — values were carried over from prior code and don't directly match the `higher_tfp_data` sheet in `cbo_ai_projections.xlsx`. |
| `acm_term_premium.csv` | Sep 2025 | NY Fed, [ACM 10-year term premium](https://www.newyorkfed.org/research/data_indicators/term-premia-tabs) |
| `primary_surplus_bea.csv` | — | BEA NIPA primary balance |
| `r_g_historic_data.xlsx` | Jan 2025 | CBO, *The Budget and Economic Outlook: 2025 to 2035* [supplemental data](https://www.cbo.gov/publication/60870), with a hand-built "master" sheet rolled up from the CBO tables |
| `r_g_master.csv` | — | Hand-assembled. Annual `r (BEA / Jared)` and `g (BEA / Jared)` columns trace to BEA NIPA; monthly `SPF 10-Year CPI`, `rgdp10`, `US 10-Year Yield (Observed)` columns trace to Philadelphia Fed SPF (10-year horizon) and FRED `DGS10`. Exact construction methodology not documented in code. |
