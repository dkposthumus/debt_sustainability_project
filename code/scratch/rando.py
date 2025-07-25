import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from fredapi import Fred
from pathlib import Path

debt = 'GFDEGDQ188S'

def get_fred_series(series_id, series_name):
    """
    Helper function to fetch a series from FRED and return it as a DataFrame.
    """
    data = fred.get_series(series_id)
    df = pd.DataFrame(data, columns=[series_name])
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    return df
fred = Fred(api_key='8905b2f5faefd705486e644f09bb8088')

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

recession_df = get_fred_series('USRECD', 'recession')
filtered_recession_df = recession_df[recession_df['date'] >= '1976-01-01']

debt = get_fred_series(debt, 'debt held by public')

# now calculate rolling 10-year level change (note data is quarterly)
debt['debt_10yr_change'] = debt['debt held by public'] - debt['debt held by public'].shift(40)

# now plot 
plt.figure(figsize=(12, 6))
sns.lineplot(data=debt, x='date', y='debt_10yr_change', label='10-year percentage point change')
plt.axhline(30, color='red', linestyle='--', label='30 ppt threshold')
add_recession_bars(plt.gca(), filtered_recession_df, shortened=False)
plt.show()