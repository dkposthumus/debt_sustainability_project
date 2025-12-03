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
code = Path.cwd() 

##################################################################
## set up fred api and relevant helper functions
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
# pull in recession data from FRED
recession_data = fred.get_series('USRECD')
# convert to DataFrame
recession_df = pd.DataFrame(recession_data, columns=['recession'])
# convert index to datetime
recession_df.index = pd.to_datetime(recession_df.index)
# reset index to have 'date' column
recession_df.reset_index(inplace=True)
recession_df.rename(columns={'index': 'date'}, inplace=True)
shortened = recession_df[recession_df['date'] >= '1962-01-01']
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
##################################################################

# i will compare my framework-produced debt paths to the historic debt path from FRED 
current_expenditures = get_fred_series('W019RCQ027SBEA', 'current_expenditures') # quarterly, billions
current_receipts = get_fred_series('W018RC1Q027SBEA', 'current_receipts') # quarterly, billions
current_interest_expenditures = get_fred_series('A091RC1Q027SBEA', 'current_interest_expenditures') # quarterly, billions
current_interest_receipts = get_fred_series('B094RC1Q027SBEA', 'current_interest_receipts') # quarterly, billions
debt_held_by_public = get_fred_series('FYGFDPUN', 'debt_held_by_public') # quarterly, millions
ngdp = get_fred_series('GDP', 'ngdp') # quarterly, billions 
growth = get_fred_series('A191RL1Q225SBEA', 'gdp_growth_rate') # quarterly, percent (real)
interest = get_fred_series('REAINTRATREARAT10Y', 'interest_rate') # monthly, percent (real)
deflator = get_fred_series('A191RD3A086NBEA', 'gdp_deflator') # quarterly, pct

df = current_expenditures.merge(current_receipts, on='date', how='outer')
df = df.merge(current_interest_expenditures, on='date', how='outer')
df = df.merge(current_interest_receipts, on='date', how='outer')
df = df.merge(debt_held_by_public, on='date', how='outer')
df = df.merge(ngdp, on='date', how='outer')
df = df.merge(growth, on='date', how='outer')
df = df.merge(interest, on='date', how='outer')
df = df.merge(deflator, on='date', how='outer')

# change gdp deflator to yoy percent change
df['gdp_deflator'] = df['gdp_deflator'].pct_change(periods=4) * 100

# convert debt_held_by_public to billions from millions
df['debt_held_by_public'] = df['debt_held_by_public'] / 1000

# now estimate total deficit 
df['total_deficit'] = df['current_expenditures'] - df['current_receipts']
df['net_interest_outlays'] = df['current_interest_expenditures'] - df['current_interest_receipts']
df['primary_deficit'] = df['total_deficit'] - df['net_interest_outlays']
for var in ['total_deficit', 'net_interest_outlays', 'primary_deficit', 'debt_held_by_public']:
    df[var + '_pct_gdp'] = (df[var] / df['ngdp']) * 100

# confirm minimum date in which all series are available
df = df.dropna(subset=['debt_held_by_public', 'total_deficit', 
                       'net_interest_outlays', 'gdp_growth_rate', 'interest_rate'])
print(f"Minimum date with all series available: {df['date'].min()}")

'''
now compute our debt snowball term, using 3 variations of the interest rate:
1. 10-year real interest rate 
2. 'sticky' 10-year real interest rate -- estimated via our law of motion 
3. net interest rate on debt held by public -- estimated via net interest outlays / debt held by public
'''
df['b_lag1'] = df['debt_held_by_public_pct_gdp'].shift(1)   # lagged debt-to-gdp ratio
sigma = 0.8
# initialize r_av to be equal to interest rate at first available date
df.loc[df.index[0], 'r_av'] = df.loc[df.index[0], 'interest_rate']
# initialize r_av for all other dates
for t in range(1, len(df)):
    r_prev = df.loc[df.index[t-1], 'r_av']
    r_curr = df.loc[df.index[t], 'interest_rate']
    r_av_curr = (sigma * r_prev) + ((1 - sigma) * r_curr)
    df.loc[df.index[t], 'r_av'] = r_av_curr
# now compute net interest rate on debt held by public
df['net_interest_rate'] = (df['net_interest_outlays'] / df['debt_held_by_public']) * 100
# make net interest rate real via fisher equation
df['net_interest_rate'] = (((1 + df['net_interest_rate'] / 100) / (1 + df['gdp_deflator'] / 100)) - 1) * 100

for var in ['interest_rate', 'r_av', 'net_interest_rate', 'gdp_growth_rate']:
    df[var] = df[var] / 100

# now compute three debt snowball terms:
df['snowball_real_10y'] = ((df['interest_rate'] - df['gdp_growth_rate']) / (1 + df['gdp_growth_rate'])) * df['b_lag1']
df['snowball_sticky_10y'] = ((df['r_av'] - df['gdp_growth_rate']) / (1 + df['gdp_growth_rate'])) * df['b_lag1']
df['snowball_net_interest'] = ((df['net_interest_rate'] - df['gdp_growth_rate']) / (1 + df['gdp_growth_rate'])) * df['b_lag1']
# now compute observed debt change, netting out primary deficit
df['debt_change_obs'] = df['debt_held_by_public_pct_gdp'] - df['b_lag1']
df['debt_change_minus_primary'] = df['debt_change_obs'] - df['primary_deficit_pct_gdp']

plt.figure(figsize=(10,6))
plt.plot(df['date'], df['debt_change_minus_primary'], 
         label='Observed Debt Change Minus Primary Deficit', color='black', linewidth=2)
plt.plot(df['date'], df['debt_change_obs'], 
         label='Observed Debt Change', color='gray', linestyle='--')
plt.plot(df['date'], df['primary_deficit_pct_gdp'], 
         label='Primary Deficit', color='red', linestyle=':')
plt.plot(df['date'], df['total_deficit_pct_gdp'], 
         label='Total Deficit', color='orange', linestyle='-.')
plt.plot(df['date'], df['net_interest_outlays'] / df['ngdp'] * 100, 
         label='Net Interest Outlays', color='blue', linestyle='-.')
plt.ylim(-10, 10)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
## add recession bars
recession_df = recession_df[recession_df['date'] >= df['date'].min()]
add_recession_bars(plt.gca(), recession_df, shortened=False)
plt.ylabel('Debt Change (% of GDP)')
plt.title('Observed Debt Dyanmics')
plt.grid(True)
plt.legend(loc='best')
plt.show()

# now plot all three as time-series
plt.figure(figsize=(10,6))
plt.plot(df['date'], df['debt_change_minus_primary'], 
         label='Observed Debt Change Minus Primary Deficit', color='black', linewidth=2)
plt.plot(df['date'], df['snowball_real_10y'], 
         label='Debt Snowball: 10-Year Real Interest Rate', color='blue', linestyle='--')
plt.plot(df['date'], df['snowball_sticky_10y'], 
         label='Debt Snowball: Sticky 10-Year Real Interest Rate', color='orange', linestyle='--')
plt.plot(df['date'], df['snowball_net_interest'], 
         label='Debt Snowball: Net Interest Rate on Debt Held by Public', color='green', linestyle='--')
plt.ylim(-10, 10)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
## add recession bars
recession_df = recession_df[recession_df['date'] >= df['date'].min()]
add_recession_bars(plt.gca(), recession_df, shortened=False)
plt.ylabel('Debt Change Minus Primary Deficit (% of GDP)')
plt.title('Estimates Debt Snowball Over Time')
plt.grid(True)
plt.legend(loc='best')
plt.show()

# now add each of these debt snowball estimates to observed primary surplus to see which best tracks observed debt change
df['debt_change_plus_primary_real_10y'] = df['snowball_real_10y'] + df['primary_deficit_pct_gdp']
df['debt_change_plus_primary_sticky_10y'] = df['snowball_sticky_10y'] + df['primary_deficit_pct_gdp']
df['debt_change_plus_primary_net_interest'] = df['snowball_net_interest'] + df['primary_deficit_pct_gdp']

plt.figure(figsize=(10,6))
plt.plot(df['date'], df['debt_change_obs'], 
         label='Observed Debt Change', color='black', linewidth=2)
plt.plot(df['date'], df['debt_change_plus_primary_real_10y'], 
         label='Debt Change Estimate: 10-Year Real Interest Rate', color='blue', linestyle='--')
plt.plot(df['date'], df['debt_change_plus_primary_sticky_10y'], 
         label='Debt Change Estimate: Sticky 10-Year Real Interest Rate', color='orange', linestyle='--')
plt.plot(df['date'], df['debt_change_plus_primary_net_interest'], 
         label='Debt Change Estimate: Net Interest Rate on Debt Held by Public', color='green', linestyle='--')
plt.ylim(-10, 10)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
## add recession bars
recession_df = recession_df[recession_df['date'] >= df['date'].min()]
add_recession_bars(plt.gca(), recession_df, shortened=False)
plt.ylabel('Debt Change (% of GDP)')
plt.title('Observed vs. Estimated Debt Change Over Time')
plt.grid(True)
plt.legend(loc='best')
plt.show()