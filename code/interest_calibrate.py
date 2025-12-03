import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fredapi import Fred
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
import numpy as np
from pathlib import Path

# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'debt_sustainability_project')
data = (work_dir / 'data')
raw_data = (data / 'briefing_piece')
output = (work_dir / 'output' / 'debt_sustainability_briefing_figures' / 'old_pdfs')
code = Path.cwd() 

# set matplotlib style
plt.style.use('mahoney_lab.mplstyle')

fred = Fred(api_key='8905b2f5faefd705486e644f09bb8088')

def get_fred_series(series_id, series_name):
    s = fred.get_series(series_id)
    df = pd.DataFrame(s, columns=[series_name])
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    return df

interest_rate_df = get_fred_series('REAINTRATREARAT10Y', 'interest_rate')
debt_held_by_public_df = get_fred_series('FYGFGDQ188S', 'debt_held_by_public')
growth_rate = get_fred_series('A191RL1Q225SBEA', 'gdp_growth_rate')

df = interest_rate_df.merge(debt_held_by_public_df, on='date', how='outer').merge(growth_rate, 
                                                                                  on='date', how='outer')

for var in ['interest_rate', 'gdp_growth_rate']:
    df[var] = df[var] / 100

df['b_lag1'] = df['debt_held_by_public'].shift(3)
df['gross_interest'] = (df['interest_rate'] / (1 + df['gdp_growth_rate'])) * df['b_lag1']
df.to_csv(raw_data / 'interest_rate_debt_growth.csv', index=False)

# now i need to find the net interest rate (net interest outlays / debt held by public)
interest_outlays = get_fred_series('A091RC1Q027SBEA', 'interest_outlays')
interest_receipts = get_fred_series('B094RC1Q027SBEA', 'interest_receipts')
debt_held_by_public_df = get_fred_series('FYGFDPUN', 'debt_held_by_public')
ngdp = get_fred_series('GDP', 'ngdp')
gdp_deflator = get_fred_series('GDPDEF', 'gdp_deflator')

net_df = interest_outlays.merge(debt_held_by_public_df, on='date', how='outer')
net_df = net_df.merge(interest_receipts, on='date', how='outer')
net_df = net_df.merge(ngdp, on='date', how='outer')
net_df = net_df.merge(gdp_deflator, on='date', how='outer')

net_df['net_interest_outlays'] = net_df['interest_outlays'] - net_df['interest_receipts']
net_df['net_interest_rate'] = net_df['net_interest_outlays'] / (net_df['debt_held_by_public'] / 1000) * 100
net_df['gdp_deflator_yoy'] = net_df['gdp_deflator'].pct_change(periods=4) * 100
# we need to make this real; use fisher formula 
net_df['real_net_interest_rate'] = (((1 + (net_df['net_interest_rate'] / 100)) / 
                                     (1 + (net_df['gdp_deflator_yoy'] / 100))) - 1) * 100

net_df['net_interest_pct_gdp'] = (net_df['net_interest_outlays'] / net_df['ngdp']) * 100
net_df['gross_interest_pct_gdp'] = (net_df['interest_outlays'] / net_df['ngdp']) * 100

plt.figure(figsize=(10,6))
plt.plot(net_df['date'], net_df['real_net_interest_rate'], label='Net Interest Rate', color='blue')
plt.plot(df['date'], df['interest_rate'] * 100, label='Gross Interest Rate', color='orange', linestyle='--')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylabel('Net Interest Rate on Debt Held by Public')
plt.title('Net Interest Rate Over Time')
plt.grid(True)
plt.legend()
plt.show()

master = pd.merge(df, net_df[['date', 'real_net_interest_rate', 
                              'net_interest_pct_gdp', 'gross_interest_pct_gdp']], on='date', how='outer')

master['net_interest_pct_gdp_check'] = ((master['real_net_interest_rate'] / (100 * (1 + master['gdp_growth_rate']))) 
                                  * master['b_lag1'])

# now compare the gross vs. net interest rate payments (as % of gdp)
plt.figure(figsize=(10,6))
graphing_df = master.dropna(subset=['gross_interest', 'net_interest_pct_gdp'])
plt.plot(graphing_df['date'], graphing_df['net_interest_pct_gdp'], 
         label='Net -- Observed', color='green')
plt.plot(graphing_df['date'], graphing_df['net_interest_pct_gdp_check'], 
         label='Net -- Model-Implied', color='blue', linestyle=':')
plt.plot(graphing_df['date'], graphing_df['gross_interest_pct_gdp'], 
         label='Gross -- Observed', color='orange')
plt.plot(graphing_df['date'], graphing_df['gross_interest'], 
         label='Gross -- Model-Implied', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylabel('Interest Payments (% of GDP)')
plt.title('Interest Payments (% of GDP) Over Time')
plt.grid(True)
plt.legend()
plt.show()