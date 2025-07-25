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

################################################################
# set up 1) matplotlib style, 2) FRED API key, and 3) recession data
################################################################
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.style.use(code / 'mahoney_lab.mplstyle')
# pull in FRED API key
fred = Fred(api_key='8905b2f5faefd705486e644f09bb8088')

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

################################################################
# define relevant functions
    # 1. add_recession_bars
    # 2. add_footnote_text
    # 3. plot_tariff_scenario
################################################################
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

def add_footnote_text(source_text):
    foot_y = 0
    plt.figtext(0.01, foot_y, "Sources:", fontfamily="Times New Roman",
                fontweight="bold", fontsize=8, ha="left", va="bottom")
    plt.figtext(0.075, foot_y, source_text, fontfamily="Times New Roman",
                fontsize=8, ha="left", va="bottom")

def plot_tariff_scenario(df, var_list, title, ylabel, plt_y0, annualized=False, yoy=False, save=False):
    plt.figure(figsize=(10, 6))
    for var in var_list:
        df[var] = pd.to_numeric(df[var], errors='coerce')
        df[var] = df[var].astype(float)
        df.dropna(subset=['date', var], inplace=True)
        if yoy:
            df[var] = (df[var] / df[var].shift(4) - 1) * 100
        if annualized:
            df[var] = ((df[var] / df[var].shift(1)) ** 4 - 1) * 100
    graphing_df = df[df['date'] >= pd.to_datetime('2025-04-01')]
    for var in var_list:
        plt.plot(graphing_df['date'], graphing_df[var], linestyle='--', 
                 label=f'{var} (Tariff Scenario)', alpha=0.8)
    plt.axvline(pd.to_datetime('2025-04-01'), color='red', linestyle='--', linewidth=0.5)
    graphing_df = df[df['date'] <= pd.to_datetime('2025-04-01')]
    plt.plot(graphing_df['date'], graphing_df[var_list[-1]], linestyle='-', 
             label='_nolegend_', alpha=0.8, color='black')
    if plt_y0:
        plt.axhline(y=0, linestyle='--', linewidth=0.75, color='black')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    add_footnote_text('Mark Zandi, Moody\'s Analytics; U.S. Treasury Department.')
    plt.tight_layout()
    if save == True:
        plt.savefig(output / f'figure_a-1.pdf')
    plt.show()

quarter_to_month = {
    1: '01',
    2: '04',
    3: '07',
    4: '10'
}

################################################################
# Figure 1: Net Government Debt vs. 10-Year Bond Yield, Advanced Economies
################################################################
countries = ['US', 'JP', 'NZ', 'AU', 'GB', 'IT', 'GR',
             'SI', 'NL', 'DK', 'CH', 'FI', 'DE', 
             'AT', 'BE', 'PT', 'CA'] # list of countries' abbreviations to pull data for
data = {}
for country in countries:
    series_id = f'IRLTLT01{country}M156N'
    try:
        data[country] = fred.get_series(series_id)
    except Exception as e:
        print(f"Error fetching data for {country}: {e}")
    # restrict to 1992 and later
for country in data:
    data[country] = data[country][data[country].index >= '1992-01-01'] # set lower bound date for 1992
# append all data into a single DataFrame
df = pd.DataFrame(data)
# convert index to datetime
df = df.reset_index()
df['index'] = pd.to_datetime(df['index'])
# rename index to 'date'
df.rename(columns={'index': 'date'}, inplace=True)
# now restrict df to only include the average for 2024 
df = df[df['date'] >= '2024-01-01']
df = df[df['date'] < '2025-01-01']
# now convert to long format 
df_long = pd.melt(df, id_vars=['date'], var_name='country', value_name='yield')
# now collapse on country 
df_long = df_long.groupby('country').mean().reset_index()
df_long = df_long[['country', 'yield']]
country_name_map = { # initialize country abbreviation - name map
    'US': 'United States',
    'JP': 'Japan',
    'NZ': 'New Zealand',
    'AU': 'Australia',
    'GB': 'United Kingdom',
    'IT': 'Italy',
    'GR': 'Greece',
    'SI': 'Slovenia',
    'NL': 'Netherlands',
    'DK': 'Denmark',
    'CH': 'Switzerland',
    'FI': 'Finland',
    'DE': 'Germany',
    'AT': 'Austria',
    'BE': 'Belgium',
    'PT': 'Portugal',
    'CA': 'Canada'
}
reverse_country_name_map = {v: k for k, v in country_name_map.items()}
# pull in govt debt data from pre-pulled IMF data 
govt_debt = pd.read_excel(raw_data / 'imf_net_debt.xls', skiprows=1)
govt_debt = govt_debt[['country', '_2024']]
govt_debt['country'] = govt_debt['country'].map(reverse_country_name_map)
# now merge two datasets 
df_long = df_long.merge(govt_debt, on='country', how='left')
# rename _2023 to 'govt_debt'
df_long.rename(columns={'_2024': 'govt_debt'}, inplace=True)
# convert govt_debt to numeric
df_long['govt_debt'] = pd.to_numeric(df_long['govt_debt'], errors='coerce')
# drop rows with NaN values in 'govt_debt' or 'yield'
df_long.dropna(subset=['govt_debt', 'yield'], inplace=True)
# convert 'yield' to numeric
df_long['yield'] = pd.to_numeric(df_long['yield'], errors='coerce')
eurozone_countries = ['IT', 'GR', 'SI', 'NL', 'FI', 'DE', 'AT', 'BE', 'PT'] # set list of abbreviatinos for eurozone
df_long['group'] = df_long['country'].map(lambda x: 'Euro zone' if x in eurozone_countries else 'G10')
palette = {'G10': '#1f77b4', 'Euro zone': '#d62728'}  # blue, red
markers = {'G10': 'D', 'Euro zone': 'D'}  # diamond markers
# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_long,
    x='govt_debt',
    y='yield',
    hue='group',
    style='group',
    palette=palette,
    markers=markers,
    s=150,  # point size
    edgecolor='black',
    alpha=1.0
)
# Add country labels next to each point
texts = []
for _, row in df_long.iterrows():
    full_name = country_name_map.get(row['country'], row['country'])
    texts.append(
        plt.text(row['govt_debt'] + 3, row['yield'] + 0.1, full_name,
                 fontsize=10, fontweight='normal'))
adjust_text(texts, 
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            expand_text=(1.05, 1.2),  # horizontal and vertical expansion factors
            only_move={'points':'y', 'text':'xy'})  # allow movement
# Customize axes and title
plt.title('Figure 1: Net Government Debt vs. 10-Year Bond Yield, Advanced Economies', 
          fontsize=14)
plt.xlabel('General government net debt, in % GDP (2024)', fontsize=12)
plt.ylabel('10-year gov\'t bond yield, in % (2024 Average)', fontsize=12)
plt.ylim(0, 5)  # Set y-axis limit
plt.xlim(-10, 250)  # Set x-axis limit
# Style tweaks
plt.legend(loc='best')
plt.grid(True, linestyle='--')
foot_y = 0
add_footnote_text("International Monetary Fund (IMF); Federal Reserve of St. Louis (FRED); Authors' analysis based on Robin Brooks, 2005.")
plt.tight_layout()
plt.savefig(output / 'figure_1.pdf')
plt.show()

################################################################
# figure 3: replicate blanchard's plot of (r-g) from survey of professional forecasters (spf)
################################################################
spf_cpi = pd.read_excel(raw_data / 'spf_cpi.xlsx')
spf_cpi.columns = spf_cpi.columns.str.lower()
# now let's convert year-quarter into datetime
# first, map quarters to months 
quarter_to_month = {
    1: '01',
    2: '04',
    3: '07',
    4: '10'
}
spf_cpi['month'] = spf_cpi['quarter'].map(quarter_to_month)
spf_cpi['date'] = pd.to_datetime(spf_cpi['year'].astype(str) + '-' + spf_cpi['month'] + '-01')
# now restrict to 1992 and later
#spf_cpi = spf_cpi[spf_cpi['date'] >= '1992-01-01']
# drop all nan values for cpi10
spf_cpi = spf_cpi.dropna(subset=['cpi10'])
# convert cpi10 to numeric
spf_cpi['cpi10'] = pd.to_numeric(spf_cpi['cpi10'], errors='coerce')
# now take median for every date 
median_cpi = spf_cpi.groupby('date')['cpi10'].median().reset_index()
# now bring in spf_gdp 
spf_gdp = pd.read_excel(raw_data / 'spf_rgdp_growth.xlsx')
spf_gdp.columns = spf_gdp.columns.str.lower()
spf_gdp['month'] = spf_gdp['quarter'].map(quarter_to_month)
spf_gdp['date'] = pd.to_datetime(spf_gdp['year'].astype(str) + '-' + spf_gdp['month'] + '-01')
#spf_gdp = spf_gdp[spf_gdp['date'] >= '1992-01-01']
# drop all nan values for rgdp10
spf_gdp = spf_gdp.dropna(subset=['rgdp10'])
spf_gdp['rgdp10'] = pd.to_numeric(spf_gdp['rgdp10'], errors='coerce')
# now take median for every date
median_gdp = spf_gdp.groupby('date')['rgdp10'].median().reset_index()
# now bring in us 10-year yield data from FRED 
us_10yr = data['US']
# make columns and convert to DataFrame
us_10yr = pd.DataFrame(us_10yr)
us_10yr = us_10yr.reset_index()
us_10yr.columns = ['date', 'us_10yr']
# now merge everything OUTER and forward fill gdp and cpi projections 
merged = pd.merge(median_cpi, median_gdp, on='date', how='outer')
merged = pd.merge(merged, us_10yr, on='date', how='outer')
# now forward fill the gdp and cpi projections
merged['cpi10'] = merged['cpi10'].ffill()
merged['rgdp10'] = merged['rgdp10'].ffill()
# now generate r variable 
merged['r']  = merged['us_10yr'] - merged['cpi10']
merged['g'] = merged['rgdp10']
merged['r-g'] = merged['r'] - merged['g']
# now collapse on the yearly average of r, g, and r-g 
merged['year'] = merged['date'].dt.year.astype(str)
collapsed = merged.groupby('year')[['r', 'g', 'r-g']].mean().reset_index()
# now convert back to datetime so we can plot it alongside recession bars 
collapsed['date'] = pd.to_datetime(collapsed['year'] + '-01-01')
# save collapsed data to csv
collapsed.to_csv(raw_data / 'r_g_clean.csv', index=False)
# Now let's plot r, g and r-g
plt.figure(figsize=(10, 6))
ax = plt.gca()  # get current axes
# Plot original series (you can also choose to plot the MA if you want)
plt.plot(collapsed['date'], collapsed['r'], 
         linestyle='-', label='r (Real 10-Year Yield)', alpha=0.8)
plt.plot(collapsed['date'], collapsed['g'],
         linestyle='-', label='g (Real Projected Growth Rate)', alpha=0.8)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
# Add recession bars
add_recession_bars(ax, recession_df, shortened=True)
# Labels and grid
plt.title('Figure 3: Real Interest Rate (r) and Real Growth Rate (g) Through 2024')
plt.ylabel('%')
plt.legend()
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
# Footnote text
add_footnote_text("Federal Reserve Bank of Philadelphia's Survey of Professional Forecasters (SPF); Federal Reserve of St. Louis (FRED); Authors' analysis.")
plt.tight_layout()
plt.savefig(output / 'figure_3.pdf')
plt.show()

################################################################
# evaluation of tariff scenarios (figures 4 and a-1)
# ################################################################
rgdp = pd.read_excel(raw_data / 'Bernstein Tariff Scenarios - Zandi - June 11, 2025.xlsx', 
                     sheet_name='Real GDP', skiprows=3)
rgdp = rgdp.dropna(subset=['S1', 'S2', 'S3'])
rgdp.rename(columns={'Unnamed: 0': 'quarter'}, inplace=True)
rgdp['year'] = rgdp['quarter'].str.extract(r'(\d{4})').astype(int)
rgdp['month'] = rgdp['quarter'].str.extract(r'[Qq](\d)')[0].astype(int).map(quarter_to_month)
rgdp['date'] = pd.to_datetime(rgdp['year'].astype(str) + '-' + rgdp['month'] + '-01')

plot_tariff_scenario(rgdp, ['S1', 'S2', 'S3'], 
                     'Annualized Real GDP Growth (%), By Tariff Scenario',
                     'Annualized Real GDP Growth (%)', plt_y0=True,
                     annualized=True)

# now make bar chart of 2025-2028 cumulative projections of RGDP growth 
scenarios = ['Scenario 1', 'Scenario 2', 'Scenario 3']
growth_rates = [2.1, 1.4, 0.4]  # as percentages
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(scenarios, growth_rates)
ax.bar_label(bars, labels=[f"{g:.1f}%" for g in growth_rates], padding=4)
ax.set_title('Figure 4: Annualized Real GDP Growth Rates Under 3 Tariff Scenarios 2025–28', pad=12)
ax.set_ylabel('Annualized Growth Rate (%)')
ax.set_ylim(0, max(growth_rates) * 1.2)  # give some headroom above the tallest bar
plt.tight_layout()
add_footnote_text('Mark Zandi, Moody\'s Analytics.')
plt.savefig(output / 'figure_4.pdf')
plt.show()

rgdp = rgdp.melt(id_vars=['date'], value_vars=['S1', 'S2', 'S3'],
                        var_name='scenario', value_name='rgdp_growth')

unemployment = pd.read_excel(raw_data / 'Bernstein Tariff Scenarios - Zandi - June 11, 2025.xlsx', 
                             sheet_name='Unemployment Rate', skiprows=2)
unemployment = unemployment.dropna(subset=['S1', 'S2', 'S3'])
unemployment.rename(columns={'Unnamed: 0': 'quarter'}, inplace=True)
unemployment['year'] = unemployment['quarter'].str.extract(r'(\d{4})').astype(int)
unemployment['month'] = unemployment['quarter'].str.extract(r'[Qq](\d)')[0].astype(float).map(quarter_to_month)
unemployment['date'] = pd.to_datetime(unemployment['year'].astype(str) + '-' + unemployment['month'] + '-01')

plot_tariff_scenario(unemployment, ['S1', 'S2', 'S3'], 
                     'Unemployment Rate Projections (%), By Tariff Scenario',
                     'Unemployment Rate (%)', plt_y0=True)
unemployment = unemployment.melt(id_vars=['date'], value_vars=['S1', 'S2', 'S3'],
                        var_name='scenario', value_name='unemployment')

pce = pd.read_excel(raw_data / 'Bernstein Tariff Scenarios - Zandi - June 11, 2025.xlsx', 
                    sheet_name='PCE Deflator', skiprows=3)
pce = pce.dropna(subset=['S1', 'S2', 'S3'])
pce.rename(columns={'Unnamed: 0': 'quarter'}, inplace=True)
pce['year'] = pce['quarter'].str.extract(r'(\d{4})').astype(int)
pce['month'] = pce['quarter'].str.extract(r'[Qq](\d)')[0].astype(float).map(quarter_to_month)
pce['date'] = pd.to_datetime(pce['year'].astype(str) + '-' + pce['month'] + '-01')

plot_tariff_scenario(pce, ['S1', 'S2', 'S3'], 
                     'Inflation Rate Projections (YoY%), By Tariff Scenario',
                     'YoY Change in PCE Deflator (%)', plt_y0 = False,
                     yoy=True)
pce = pce.melt(id_vars=['date'], value_vars=['S1', 'S2', 'S3'],
                        var_name='scenario', value_name='pce')

tbill = pd.read_excel(raw_data / 'Bernstein Tariff Scenarios - Zandi - June 11, 2025.xlsx', 
                      sheet_name='10-Yr T-Yield', skiprows=2)
tbill = tbill.dropna(subset=['S1', 'S2', 'S3'])
tbill.rename(columns={'Unnamed: 0': 'quarter'}, inplace=True)
tbill['year'] = tbill['quarter'].str.extract(r'(\d{4})').astype(int)
tbill['month'] = tbill['quarter'].str.extract(r'[Qq](\d)')[0].astype(float).map(quarter_to_month)
tbill['date'] = pd.to_datetime(tbill['year'].astype(str) + '-' + tbill['month'] + '-01')

plot_tariff_scenario(tbill, ['S1', 'S2', 'S3'], 
                     'Figure A-1: 10-Year Treasury Yield (%), By Tariff Scenario',
                     '10-Year Treasury Yield (%)', plt_y0 = False, save=True)
tbill = tbill.melt(id_vars=['date'], value_vars=['S1', 'S2', 'S3'],
                        var_name='scenario', value_name='tbill')

# now merge all the master projections 
master_projections = pd.merge(tbill, pce, on=['scenario', 'date'], how='outer')
master_projections = pd.merge(master_projections, rgdp, 
                              on=['scenario', 'date'], how='outer')
graphing_df = master_projections[master_projections['date'] >= pd.to_datetime('2025-04-01')]

# now let's estimate projected r (tbill - pce)
graphing_df['r'] = graphing_df['tbill'] - graphing_df['pce']
graphing_df['g'] = graphing_df['rgdp_growth']
graphing_df['r-g'] = graphing_df['r'] - graphing_df['g']

################################################################
# figure 5: plot tbl projections
#################################################################
tbl = pd.read_excel(raw_data / 'tbl_tariffs_projections.xlsx', skiprows=4, sheet_name='F3')
# cap at 10/1/2028
tbl['Date'] = pd.to_datetime(tbl['Date'])
tbl = tbl[tbl['Date'] <= pd.to_datetime('2028-10-01')]
# convert 'All 2025 Tariffs to Date' to numeric
plt.figure(figsize=(10, 6))
plt.plot(tbl['Date'], tbl['All 2025 Tariffs to Date'], 
         label='All 2025 Tariffs to Date', linestyle='-', linewidth=2, color='blue')
plt.axhline(y=0, linestyle='--', linewidth=0.75, color='black')
plt.ylabel('Real GDP Level Effects')
add_footnote_text('The Budget Lab, "State of U.S. Tariffs: June 17, 2025".')
plt.title('Figure 5: Real GDP Level Effects of Tariffs in 2025, Relative to Baseline')
plt.tight_layout()
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
plt.savefig(output / 'figure_5.pdf')
plt.show()

################################################################
# figure 6: plot 10-year premium (Haver)
# ################################################################
yr10_prem = pd.read_excel(raw_data / 'haver_10yr_premium.xlsx', skiprows=4)
# now we want to rename columns 
yr10_prem.rename(columns={'.SOURCE': 'date',
                          'FRBNY': '10yr_premium'}, inplace=True)
# convert date to datetime
yr10_prem['date'] = pd.to_datetime(
    yr10_prem['date'],
    format='%d-%b-%y',   # day–abbrev month–2-digit year
    errors='raise'       # will error if any string doesn’t match
)
# now keep only those in 2005 or later 
yr10_prem = yr10_prem[yr10_prem['date'] >= pd.to_datetime('2005-01-01')]
# collapse on monthly average 
yr10_prem['month'] = yr10_prem['date'].dt.to_period('M')
yr10_prem = yr10_prem.groupby('month')['10yr_premium'].mean().reset_index()
# now convert 'month' back to datetime
yr10_prem['date'] = yr10_prem['month'].dt.to_timestamp()
# now plot 
plt.figure(figsize=(10, 6))
plt.plot(yr10_prem['date'], yr10_prem['10yr_premium'],
         label='10-Year Treasury Yield Premium', linestyle='-', color='blue')
plt.axhline(y=0, linestyle='--', linewidth=0.75, color='black')
plt.ylabel('10-Year Treasury Yield Premium (%)')
plt.title('Figure 6: 10-Year Treasury Yield Premium Over Time')
filtered_recession = recession_df[recession_df['date'] >= pd.to_datetime('2005-01-01')]
add_recession_bars(plt.gca(), filtered_recession, shortened=True)
add_footnote_text('Haver Analytics; Federal Reserve Bank of New York.')
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig(output / 'figure_6.pdf')
plt.show()

################################################################
# figures 7 and 8: plot GS projections of real interest expenses
# ################################################################
df = pd.read_csv(raw_data / 'gs_projections_cleaned_2.csv')
# The first column will be called something like 'Unnamed: 0'—rename it to “year”
df = df.rename(columns={df.columns[0]:'year'})
# now convert year into datetime
df['year'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
# now convert all other columns to numeric
# first we're just plotting federal real interest expense 
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['Actual'],
            label='Actual', color='blue', linestyle='-')
plt.plot(df['year'], df['5% Interest Rate Scenario'],
            label='5% Interest Rate Scenario', color='purple', linestyle='--')
plt.plot(df['year'], df['4% Interest Rate (GS Baseline)'],
            label='4% Interest Rate Scenario (GS Baseline)', color='orange', linestyle='--')
plt.plot(df['year'], df['3% Interest Rate Scenario'],
         label='3% Interest Rate Scenario', color='green', linestyle='--')
plt.plot(df['year'], df['2% Interest Rate Scenario'],
            label='2% Interest Rate Scenario', color='red', linestyle='--')
plt.ylabel('% of GDP')
plt.title('Figure 7: Debt Held by the Public (% of GDP)')
plt.legend(loc='best')
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7)
filtered_recession_df = recession_df[recession_df['date'] >= pd.to_datetime('1941-01-01')]
add_recession_bars(plt.gca(), filtered_recession_df, shortened=False)
add_footnote_text('Abecasis et al, Goldman Sachs Global Investment Research.')
plt.tight_layout()
plt.savefig(output / 'figure_7.pdf')
plt.show()

df = pd.read_csv(raw_data / 'gs_projections_cleaned.csv')
# The first column will be called something like 'Unnamed: 0'—rename it to “year”
df = df.rename(columns={df.columns[0]:'year'})
# now convert year into datetime
df['year'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
# now convert all other columns to numeric
# first we're just plotting federal real interest expense 
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['Actual'],
            label='Actual', color='blue', linestyle='-')
plt.plot(df['year'], df['5% Interest Rate Scenario'],
            label='5% Interest Rate Scenario', color='purple', linestyle='--')
plt.plot(df['year'], df['4% Interest Rate (GS Baseline)'],
            label='4% Interest Rate Scenario (GS Baseline)', color='orange', linestyle='--')
plt.plot(df['year'], df['3% Interest Rate Scenario'],
         label='3% Interest Rate Scenario', color='green', linestyle='--')
plt.plot(df['year'], df['2% Interest Rate Scenario'],
            label='2% Interest Rate Scenario', color='red', linestyle='--')
plt.ylabel('% of GDP')
plt.title('Figure 8: Federal Real Interest Expense Projections (% of GDP)')
plt.axhline(y=0, linestyle='--', linewidth=0.75, color='black')
plt.legend(loc='upper left')
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7)
add_recession_bars(plt.gca(), filtered_recession_df, shortened=False)
add_footnote_text('Abecasis et al, Goldman Sachs Global Investment Research.')
plt.tight_layout()
plt.savefig(output / 'figure_8.pdf')
plt.show()

################################################################
# figure 9 (each version): Primary Deficit, Historical Data and Projections (% of GDP)
################################################################
# Load and clean
primary_defs = pd.read_excel(raw_data / 'primary_def_projections.xlsx', skiprows=4)
primary_defs.rename(columns={'.SOURCE': 'year'}, inplace=True)
primary_defs['year'] = primary_defs['year'].astype(int)
primary_defs['year4'] = primary_defs['year'].apply(lambda y: 1900 + y if y >= 65 else 2000 + y)
primary_defs['date'] = pd.to_datetime(primary_defs['year4'].astype(str) + '-01-01')
# Filter post-2000
primary_defs = primary_defs[primary_defs['date'] >= pd.to_datetime('1965-01-01')]
# Split data
historical = primary_defs[primary_defs['date'] <= pd.to_datetime('2025-01-01')]
projections = primary_defs[primary_defs['date'] >= pd.to_datetime('2025-01-01')]
# Main plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(historical['date'], historical['CBO/H'], 
        label='CBO Historical Primary Deficit', linestyle='-', color='blue')
ax.plot(projections['date'], projections['CBO (Senate - Permanent)'], 
        label='Pre-GOP Policy Bill Deficit Forecast (CBO)', linestyle='--', color='blue')
ax.plot(projections['date'], projections['Budget Lab (Senate - Permanent)'], 
        label='Post-Senate GOP Policy Bill Deficit Forecast (Budget Lab)', linestyle='--', 
        color='orange')
# Add orange dot annotations
annotations = {2025: 0.5, 2029: 1, 2035: 2.5}
annotation_years = [2025, 2029, 2035]
for year, offset in annotations.items():
    x = pd.Timestamp(f'{year}-01-01')
    y = offset
    ax.scatter(x, y, color='red', s=50, zorder=10)
# Add horizontal line, labels, grid
ax.axhline(y=0, linestyle='--', linewidth=0.75, color='black')
ax.set_ylabel('Primary Deficit (% of GDP)')
ax.set_title('Figure 9: Primary Deficit, Historical Data and Projections (% of GDP)')
shortened_recession_df = recession_df[recession_df['date'] >= pd.to_datetime('1965-01-01')]
add_recession_bars(ax, shortened_recession_df, shortened=False)
fig.subplots_adjust(bottom=0.85)
caption = (
    "Congressional Budget Office (CBO); The Budget Lab analysis; "
    "Abecasis et al, Goldman Sachs Economics Research.\n"
    "Red dots denote the primary surplus needed to keep real interest "
    "expenses below 2 % of GDP according to Goldman Sachs Economics Research.\n"
    "The Post-Senate GOP Policy Bill Deficit Forecast is based on the Budget "
    "Lab's analysis of the Senate GOP's policy bill if enacted permanently.\n"
    'Budget Lab, "The Financial Cost of the Senate Budget Bill’s Tax Provisions".'
)
fig.text(0.01, 0.0, caption, fontsize=9, va="top", ha="left")
ax.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
ax.legend(loc='lower left')
# === Inset Plot ===
ax_inset = inset_axes(
    ax,
    width="45%", height="35%",
    bbox_to_anchor=(0.80, 0.08, 0.95, 0.95),  # (x0, y0, width, height)
    bbox_transform=ax.transAxes,
    loc='upper left',
    borderpad=0
)
forecast = projections[projections['date'].dt.year <= 2035]  # limit inset to 2025–2035
ax_inset.plot(forecast['date'], forecast['CBO (Senate - Permanent)'], label='CBO', color='blue', 
              linestyle = '--')
ax_inset.plot(forecast['date'], forecast['Budget Lab (Senate - Permanent)'], 
              label='TBL', color='orange', 
              linestyle = '--')
# Add same dots in inset
annotation_years = [2025, 2029, 2031]
for year, offset in annotations.items():
    x = pd.Timestamp(f'{year}-01-01')
    y = offset
    ax_inset.scatter(x, y, color='red', s=50, zorder=10)
    ax_inset.text(x - pd.Timedelta(days=90), 
                      y + 0.25, 
                      f"{offset:.1f}%", ha='center', va='bottom', color='red',
                fontweight='bold', zorder=11)
ax_inset.set_title('Primary Deficits: Forecasts', fontsize=10)
ax_inset.tick_params(labelsize=8)
ax_inset.axhline(y=0, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
ax_inset.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
# set x lower limit to 2024
ax_inset.set_xlim(pd.to_datetime('2024-01-01'), pd.to_datetime('2035-06-01'))
ax_inset.set_ylim(-4, 3.7)
plt.tight_layout()
plt.savefig(output / 'figure_9_alt.pdf', bbox_inches='tight')
plt.show()

# now let's break out the two separately:
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.08, bottom=0.08)
# Plot CBO Historical Primary Deficit
plt.plot(historical['date'], historical['CBO/H'],
         label='CBO Historical Primary Deficit', linestyle='-', color='blue')
# Plot CBO Pre-GOP Policy Bill Deficit Forecast
plt.plot(projections['date'], projections['CBO (Senate - Permanent)'],
         label='Pre-GOP Policy Bill Deficit Forecast (CBO)', linestyle='--', color='blue')
# Plot Budget Lab Post-Senate GOP Policy Bill Deficit Forecast
plt.plot(projections['date'], projections['Budget Lab (Senate - Permanent)'],
         label='Post-Senate GOP Policy Bill Deficit Forecast (Budget Lab)', linestyle='--', 
         color='orange')
# add red dot annotations 
plt.scatter(pd.to_datetime('2025-01-01'), 0.5, color='red', s=50, zorder=10)
plt.scatter(pd.to_datetime('2029-01-01'), 1, color='red', s=50, zorder=10)
plt.scatter(pd.to_datetime('2035-01-01'), 2.5, color='red', s=50, zorder=10)
# Add text annotations for red dots
plt.text(pd.to_datetime('2025-01-01') - pd.Timedelta(days=90), 
         0.5 + 0.25, '0.5%', ha='center', va='bottom',
         fontweight='bold', color='red', zorder=11)
plt.text(pd.to_datetime('2029-01-01') - pd.Timedelta(days = 90),
            1 + 0.25, '1.0%', ha='center', va='bottom',
            fontweight='bold', color='red', zorder=11)
plt.text(pd.to_datetime('2035-01-01') - pd.Timedelta(days = 90),
            2.5 + 0.25, '2.5%', ha='center', va='bottom',
            fontweight='bold', color='red', zorder=11)
# Add horizontal line at 0
plt.axhline(y=0, linestyle='--', linewidth=0.75, color='black')
# Add recession bars
add_recession_bars(plt.gca(), shortened_recession_df, shortened=False)
# Add labels and title
plt.ylabel('Primary Deficit (% of GDP)')
plt.title('Figure 9: Primary Deficit, Historical Data and Projections (% of GDP)')
fig.text(
    0.02,           # x-pos in figure fraction
    0,           # y-pos (a bit above bottom edge)
    caption,
    transform=fig.transFigure,
    ha="left", va="bottom",
    fontsize=8, wrap=True, 
)
# Add legend
plt.legend(loc='lower left')
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout(rect=[0.00, 0.08, 1, 1])
plt.savefig(output / 'figure_9.pdf')
plt.show()

################################################################
# figure a-2: illustrative debt dynamics: r < g vs r > g
################################################################
# Parameters
n_years = 30 # 30-year time horizon
b0 = 1.0 # starting debt as a percentage of GDP for year = 1

# Define two scenarios
scenarios = {
    "r > g (deficit)": {"r": 0.04, "g": 0.02, 'd': -0.02},
    "r < g (deficit)": {"r": 0.02, "g": 0.03, 'd': -0.02},
    "r > g (surplus)": {"r": 0.04, "g": 0.02, 'd': 0.02},
    "r < g (surplus)": {"r": 0.02, "g": 0.03, 'd': 0.02},
}

# Simulate debt paths
results = {}
table_rows = []
for i, (label, params) in enumerate(scenarios.items(), start=1):
    r, g, d = params["r"], params["g"], params["d"]
    b_path = np.zeros(n_years) # initialize debt path array
    b_path[0] = b0 # set initial value for debt path
    b_path = b0 * ((1 + r) / (1 + g)) ** np.arange(n_years) \
         - d * (1 - ((1 + r)/(1 + g)) ** np.arange(n_years)) \
           / (1 - (1 + r)/(1 + g)) # vectorized calculation of debt path
    results[label] = b_path * 100 # convert to percentage (%)

    # Append row to table data
    table_rows.append([f'({i})', f"{r:.0%}", f"{g:.0%}", f"{d:.0%}"]) # append these values

fig, ax = plt.subplots(figsize=(10, 6))
# Plot lines and label them at chosen x positions
for i, (label, path) in enumerate(results.items(), start=1):
    line, = ax.plot(range(n_years), path, label=f"({i}) {label}")
    x_pos = 15
    if i == 1:
        offset = 25 # we have to do a custom offset because of the steepness of first line
    else:
        offset = 8
    y_pos = path[x_pos] + offset
    ax.text(
        x_pos + 0.5,  # small rightward offset for labeling cleanliness / neatness
        y_pos,
        f"({i}) {label}",
        color=line.get_color(),
        fontsize=10,
        va='center'
    )
ax.axhline(100, color='gray', linestyle='--') # Horizontal reference line
# Add table
table_data = [['', 'r', 'g', 's']] + table_rows
table = plt.table(
    cellText=table_data,
    colWidths=[0.1] * 4,
    cellLoc='center',
    loc='upper left',
    bbox=[0.05, 0.65, 0.15, 0.3],
)
table.scale(1, 1.5)
# Titles and labels
ax.set_title("Figure A-2: Illustrative Debt Dynamics: r < g vs r > g", fontsize=14, weight='bold')
ax.set_xlabel("Years", fontsize=12)
ax.set_ylabel("Debt, in % of GDP", fontsize=12)
ax.grid(True)
add_footnote_text("Authors' analysis.")
plt.tight_layout()
plt.savefig(output / 'figure_a-2.pdf')
plt.show()

################################################################
# extra graph: plot (r-g) from observed data / Jared's Calculations 
################################################################
r_g = pd.read_excel(raw_data / 'r g Historical-Budget-Data.xlsx', 
                    sheet_name='3. Outlays r-g', skiprows=8)
r_g.rename(columns={'Unnamed: 0': 'year'}, inplace=True)
# force conversion to numeric for vars 
for var in ['r', 'g']:
    r_g[var] = pd.to_numeric(r_g[var], errors='coerce')
    # drop rows with NaN values
r_g.dropna(subset=['year', 'r', 'g'], inplace=True)
for var in ['r', 'g']:
    r_g[var] = r_g[var].astype(float)

# restrict to 1992 and later 
for var in ['r', 'g', 'r-g']:
    r_g[var] = r_g[var] * 100
    r_g[f'{var}_ma10'] = r_g[var].rolling(window=10).mean()
r_g['date'] = pd.to_datetime(r_g['year'].astype(str) + '-01-01')
#r_g = r_g[r_g['year'] >= 1992]

plt.figure(figsize=(10, 6))
ax = plt.gca()  # Get the current axes to pass into functions
plt.plot(r_g['date'], r_g['r'], linestyle='-', 
    label='r (Nominal Net Interest Rate)')
plt.plot(r_g['date'], r_g['g'], linestyle='-', 
    label='g (Nominal Growth Rate)')
plt.plot(r_g['date'], r_g['r'] - r_g['g'], linestyle='-', 
    label='r-g')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
# Add recession bars using function:
add_recession_bars(ax, recession_df, shortened=True)
plt.title('Real Interest Rate (r) and Real Growth Rate (g)')
plt.ylabel('Percentage')
plt.legend()
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
# Add footnote text using function:
add_footnote_text("Bureau of Economic Analysis (BEA), Authors' analysis.")
plt.tight_layout()
plt.close()

################################################################
# extra graph: compare our and Blanchard's estimate of (r-g)
# ################################################################
plt.figure(figsize=(10, 6))
ax = plt.gca()  # get current axes
cmap = plt.get_cmap('tab10')
# Plot r-g from observed BEA data
plt.plot(r_g['date'], r_g['r-g'],
         linestyle='-', label='r-g (Observed BEA Data)', alpha=0.8, color=cmap(0))
# Plot r-g from SPF projections
plt.plot(merged['date'], merged['r-g'],
         linestyle='-', label='r-g (SPF Projections)', alpha=0.8, color=cmap(1))
# Add horizontal line at 0
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
# Add recession bars
add_recession_bars(ax, recession_df, shortened=True)
# Labels and grid
plt.title('Comparison of r-g Estimates')
plt.ylabel('Percentage')
plt.legend()
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
# Footnote text
add_footnote_text("Bureau of Economic Analysis (BEA), Federal Reserve Bank of Philadelphia, Federal Reserve of St. Louis (FRED), Authors' analysis.")
plt.tight_layout()
plt.close()

filtered_r_g = r_g[['r', 'g', 'r-g', 'date']]
filtered_r_g.rename(
    columns = {
        'r': 'r (BEA / Jared)',
        'g': 'g (BEA / Jared)',
        'r-g': 'r-g (BEA / Jared)'
    }, inplace=True
)
filter_merged = merged[['date', 'cpi10', 'rgdp10', 'us_10yr',
                        'r', 'g', 'r-g']]
filter_merged.rename(
    columns = {
        'cpi10': 'SPF 10-Year CPI',
        'gdp10': 'SPF Real GDP Growth',
        'us_10yr': 'US 10-Year Yield (Observed)',
        'r': 'r (SPF / Blanchard)',
        'g': 'g (SPF / Blanchard)',
        'r-g': 'r-g (SPF / Blanchard)'
    }, inplace=True
)
df = pd.merge(filtered_r_g, filter_merged, on='date',
              how='outer')
df.to_csv('r_g_master.csv', index=False)

################################################################
# extra graph: using tariff forecasts to plot r-g forecasts
# ################################################################
plt.figure(figsize=(10, 6))
ax = plt.gca()  # get current axes

# plot r-g from Jared's / BEA's projections
r_g_filtered = r_g[r_g['date'] >= pd.to_datetime('2016-01-01')]
plt.plot(r_g_filtered['date'], r_g_filtered['r-g'],
         linestyle='-', label='r-g (Observed BEA Data)', alpha=0.8)
# Plot r-g from SPF projections
merged_filtered = merged[merged['date'] >= pd.to_datetime('2016-01-01')]
merged_filtered['year'] = merged_filtered['date'].dt.year.astype(str)
merged_filtered = merged_filtered.groupby('year')['r-g'].mean().reset_index()
merged_filtered['date'] = pd.to_datetime(
    merged_filtered['year'] + '-' + '01' + '-' + '01'
)

plt.plot(merged_filtered['date'], merged_filtered['r-g'],
         linestyle='-', label='r-g (SPF Projections)', alpha=0.8)

# now plot r-g from the 3 tariff scenarios 
for scenario, label in zip(['S1', 'S2', 'S3'], ['S1', 'S2', 'S3']):
    scenario_df = graphing_df[graphing_df['scenario'] == scenario]
    plt.plot(scenario_df['date'], scenario_df['r-g'], 
             label=f'r-g (Tariff {scenario} Scenario)', linestyle='--')

plt.axvline(pd.to_datetime('2025-04-01'), color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, linestyle='--', color='black', linewidth=0.75)
plt.ylabel('r - g (%)')
plt.legend(loc='best')
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
plt.title('Projections of (r-g) Under Different Tariff Scenarios')
plt.show()

################################################################
# extra graph: plot deficit and unemployment rate 
# ################################################################
# pull in deficit (as % of GDP) data from FRED
deficit_data = fred.get_series('FYFSGDA188S')
unemployment_data = fred.get_series('UNRATE')
# convert to DataFrame
deficit_df = pd.DataFrame(deficit_data, columns=['deficit'])
unemployment_df = pd.DataFrame(unemployment_data, columns=['unemployment'])
# convert index to datetime
deficit_df.index = pd.to_datetime(deficit_df.index)
unemployment_df.index = pd.to_datetime(unemployment_df.index)
# reset index to have 'date' column
deficit_df.reset_index(inplace=True)
unemployment_df.reset_index(inplace=True)
deficit_df.rename(columns={'index': 'date'}, inplace=True)
unemployment_df.rename(columns={'index': 'date'}, inplace=True)
# merge 
df = pd.merge(deficit_df, unemployment_df, on='date', how='outer')
# now forward fill deficit data 
df['deficit'] = df['deficit'].ffill()
# now multiply deficit by -1
df['deficit'] = df['deficit'] * -1
# restrict to 1962 and later 
df = df[df['date'] >= '1962-01-01']
# now let's collapse on year 
df['year'] = df['date'].dt.year
df = df.groupby('year')[['date', 'deficit', 'unemployment']].mean().reset_index()
# we're going to be plotting two y-axes
plt.figure(figsize=(10, 6))
ax1 = plt.gca()  # Get the current axes
# Plot deficit on primary y-axis
ax1.plot(df['date'], df['deficit'], label='Deficit (% of GDP) (Left Axis)', 
         linestyle='-', linewidth=2, color='blue')
ax1.set_ylabel('Deficit (% of GDP)')
ax1.tick_params(axis='y')
# Create a secondary y-axis for unemployment
ax2 = ax1.twinx()
# Plot unemployment on secondary y-axis
ax2.plot(df['date'], df['unemployment'], label='Unemployment Rate (%) (Right Axis)', 
         linestyle='--', linewidth=2, color='red')
ax2.set_ylabel('Unemployment Rate (%)')
ax2.tick_params(axis='y')
# Add recession 
add_recession_bars(ax1, recession_df)
# Add title and grid
plt.title('US Deficit (% of GDP) and Unemployment Rate Over Time')
plt.grid(axis='both', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
# Add footnote text
add_footnote_text("Federal Reserve of St. Louis (FRED), Authors' analysis.")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# Combine and add legend
plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
plt.tight_layout()
plt.show()