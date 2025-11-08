import pandas as pd
import matplotlib.pyplot as plt

# Leading indicators data
zero_curve_df = pd.read_excel('zero_curve.xlsx', sheet_name='Daily')
interest_rate_df = pd.read_excel('interest_rates_spread.xlsx', sheet_name='Daily')
pmi_df = pd.read_excel('PMI.xlsx')

# Load the unemployment data from the 'Monthly' sheet
unemploy_df = pd.read_excel('unemployment.xlsx', sheet_name='Monthly')
CPI_df = pd.read_excel('CPI_for_service.xlsx', sheet_name='Monthly')
commerce_df = pd.read_excel('commercial_loan.xlsx')

# Calculate monthly average for zero curve data (THREEFY1 column)
zero_curve_df['observation_date'] = pd.to_datetime(zero_curve_df['observation_date'])
zero_curve_df['month'] = zero_curve_df['observation_date'].dt.to_period('M')
monthly_avg = zero_curve_df.groupby('month')['THREEFY1'].mean().reset_index()
monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()

# Calculate monthly average for interest rates spread
interest_rate_df['observation_date'] = pd.to_datetime(interest_rate_df['observation_date'])
interest_rate_df['month'] = interest_rate_df['observation_date'].dt.to_period('M')
interest_rate_monthly_avg = interest_rate_df.groupby('month').mean().reset_index()
interest_rate_monthly_avg['month'] = interest_rate_monthly_avg['month'].dt.to_timestamp()

from sklearn.preprocessing import StandardScaler

# --- Prepare monthly averages for each indicator using already loaded DataFrames ---
# Interest Rate Spread
interest_rate_df['month'] = interest_rate_df['observation_date'].dt.to_period('M')
interest_rate_monthly = interest_rate_df.groupby('month').mean().reset_index()
interest_rate_monthly['month'] = interest_rate_monthly['month'].dt.to_timestamp()

# PMI
pmi_df['month'] = pmi_df['Date'].dt.to_period('M')
pmi_monthly = pmi_df.groupby('month')['Index'].mean().reset_index()
pmi_monthly['month'] = pmi_monthly['month'].dt.to_timestamp()

# Zero Curve
zero_curve_df['month'] = zero_curve_df['observation_date'].dt.to_period('M')
zero_curve_monthly = zero_curve_df.groupby('month')['THREEFY1'].mean().reset_index()
zero_curve_monthly['month'] = zero_curve_monthly['month'].dt.to_timestamp()

# --- Merge all indicators on month ---
merged = interest_rate_monthly.merge(pmi_monthly, on='month', how='inner').merge(zero_curve_monthly, on='month', how='inner')

# --- Standardize each indicator one by one ---
scaler = StandardScaler()
merged['InterestRate_std'] = scaler.fit_transform(merged[[interest_rate_monthly.columns[1]]])
merged['PMI_std'] = scaler.fit_transform(merged[['Index']])
merged['THREEFY1_std'] = scaler.fit_transform(merged[['THREEFY1']])

# --- Calculate combined leading indicator ---
merged['leading_indicator'] = merged[['InterestRate_std','PMI_std','THREEFY1_std']].mean(axis=1)

# --- Filter for the past 2 years ---
mask = (merged['month'] >= '2019-01-01') & (merged['month'] <= '2025-09-30')
trend = merged.loc[mask]


# Unemployment
unemploy_df['month'] = unemploy_df['observation_date'].dt.to_period('M')
unemploy_monthly = unemploy_df.groupby('month')['UEMPMEAN'].mean().reset_index()
unemploy_monthly['month'] = unemploy_monthly['month'].dt.to_timestamp()

# CPI
CPI_df['month'] = CPI_df['observation_date'].dt.to_period('M')
CPI_monthly = CPI_df.groupby('month')['CUSR0000SAS'].mean().reset_index()
CPI_monthly['month'] = CPI_monthly['month'].dt.to_timestamp()

# Commercial Loan
commerce_df['month'] = commerce_df['date'].dt.to_period('M')
commerce_monthly = commerce_df.groupby('month')['commericial_loan'].mean().reset_index()
commerce_monthly['month'] = commerce_monthly['month'].dt.to_timestamp()

# --- Merge all lagging indicators on month ---
lagging_merged = unemploy_monthly.merge(CPI_monthly, on='month', how='inner').merge(commerce_monthly, on='month', how='inner')

# --- Standardize each lagging indicator one by one ---
scaler = StandardScaler()
lagging_merged['Unemployment_std'] = scaler.fit_transform(lagging_merged[['UEMPMEAN']])
lagging_merged['CPI_std'] = scaler.fit_transform(lagging_merged[['CUSR0000SAS']])
lagging_merged['LoanAmount_std'] = scaler.fit_transform(lagging_merged[['commericial_loan']])

# --- Calculate combined lagging indicator ---
lagging_merged['lagging_indicator'] = lagging_merged[['Unemployment_std','CPI_std','LoanAmount_std']].mean(axis=1)

# --- Filter for the past 2 years ---
mask = (lagging_merged['month'] >= '2019-01-01') & (lagging_merged['month'] <= '2025-09-30')
lagging_trend = lagging_merged.loc[mask]

from scipy.interpolate import make_lsq_spline, BSpline
import numpy as np
import pandas as pd

start_date = pd.to_datetime("2023-10-01")
end_date = pd.to_datetime("2025-09-01")

trend_sorted = trend[(trend['month'] >= start_date) & (trend['month'] <= end_date)].sort_values('month')
x_lead = np.arange(len(trend_sorted))
y_lead = trend_sorted['leading_indicator'].values

# Choose knots: 5 internal knots evenly spaced (excluding endpoints)
num_knots = 5
knots_lead = np.linspace(x_lead[1], x_lead[-2], num_knots)
t_lead = np.concatenate(([x_lead[0]]*4, knots_lead, [x_lead[-1]]*4))
spl_lead = make_lsq_spline(x_lead, y_lead, t_lead, k=3)
x_lead_dense = np.linspace(x_lead.min(), x_lead.max(), 300)
y_lead_smooth = spl_lead(x_lead_dense)
months_lead_dense = pd.date_range(trend_sorted['month'].iloc[0], trend_sorted['month'].iloc[-1], periods=300)

# --- BSpline for lagging indicator ---
lagging_trend_sorted = lagging_trend[(lagging_trend['month'] >= start_date) & (lagging_trend['month'] <= end_date)].sort_values('month')
x_lag = np.arange(len(lagging_trend_sorted))
y_lag = lagging_trend_sorted['lagging_indicator'].values
knots_lag = np.linspace(x_lag[1], x_lag[-2], num_knots)
t_lag = np.concatenate(([x_lag[0]]*4, knots_lag, [x_lag[-1]]*4))
spl_lag = make_lsq_spline(x_lag, y_lag, t_lag, k=3)
x_lag_dense = np.linspace(x_lag.min(), x_lag.max(), 300)
y_lag_smooth = spl_lag(x_lag_dense)
months_lag_dense = pd.date_range(lagging_trend_sorted['month'].iloc[0], lagging_trend_sorted['month'].iloc[-1], periods=300)

# --- Plot both smoothed indicators ---
plt.figure(figsize=(12, 6))
plt.plot(trend_sorted['month'], trend_sorted['leading_indicator'], 'o', label='Leading Indicator')
plt.plot(months_lead_dense, y_lead_smooth, color='blue', linestyle='-.', linewidth = 3)
plt.plot(lagging_trend_sorted['month'], lagging_trend_sorted['lagging_indicator'], 'o', color='orange', label='Lagging Indicator')
plt.plot(months_lag_dense, y_lag_smooth, '-', color='orange', linewidth = 3)
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Indicator Value (Standardized)')
plt.title('Smoothed Leading and Lagging Indicators over past 2 years')
plt.legend()
plt.tight_layout()
plt.savefig('Q2_leading_lagging_indicators_2yrs.png', dpi = 300)
plt.show()

start_date = '2020-03-01'
end_date = '2024-03-31'
trend_sorted = trend[(trend['month'] >= start_date) & (trend['month'] <= end_date)].sort_values('month')
x_lead = np.arange(len(trend_sorted))
y_lead = trend_sorted['leading_indicator'].values

# Choose knots: 5 internal knots evenly spaced (excluding endpoints)
num_knots = 5
knots_lead = np.linspace(x_lead[1], x_lead[-2], num_knots)
t_lead = np.concatenate(([x_lead[0]]*4, knots_lead, [x_lead[-1]]*4))
spl_lead = make_lsq_spline(x_lead, y_lead, t_lead, k=3)
x_lead_dense = np.linspace(x_lead.min(), x_lead.max(), 300)
y_lead_smooth = spl_lead(x_lead_dense)
months_lead_dense = pd.date_range(trend_sorted['month'].iloc[0], trend_sorted['month'].iloc[-1], periods=300)

# --- BSpline for lagging indicator ---
lagging_trend_sorted = lagging_trend[(lagging_trend['month'] >= start_date) & (lagging_trend['month'] <= end_date)].sort_values('month')
x_lag = np.arange(len(lagging_trend_sorted))
y_lag = lagging_trend_sorted['lagging_indicator'].values
knots_lag = np.linspace(x_lag[1], x_lag[-2], num_knots)
t_lag = np.concatenate(([x_lag[0]]*4, knots_lag, [x_lag[-1]]*4))
spl_lag = make_lsq_spline(x_lag, y_lag, t_lag, k=3)
x_lag_dense = np.linspace(x_lag.min(), x_lag.max(), 300)
y_lag_smooth = spl_lag(x_lag_dense)
months_lag_dense = pd.date_range(lagging_trend_sorted['month'].iloc[0], lagging_trend_sorted['month'].iloc[-1], periods=300)

# --- Plot both smoothed indicators ---
plt.figure(figsize=(12, 6))
plt.plot(trend_sorted['month'], trend_sorted['leading_indicator'], 'o', label='Leading Indicator')
plt.plot(months_lead_dense, y_lead_smooth, color='blue', linestyle='-.', linewidth = 3)
plt.plot(lagging_trend_sorted['month'], lagging_trend_sorted['lagging_indicator'], 'o', color='orange', label='Lagging Indicator')
plt.plot(months_lag_dense, y_lag_smooth, '-', color='orange', linewidth = 3)
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Indicator Value (Standardized)')
plt.title('Leading and Lagging Indicators over COVID period')
plt.legend()
plt.tight_layout()
plt.savefig('Q2_leading_lagging_indicators_COVID.png', dpi = 300)
plt.show()
