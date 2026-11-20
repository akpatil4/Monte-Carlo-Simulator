import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Portfolio Risk Simulator", layout="wide")
st.title("Monte Carlo Portfolio Risk Simulator")

# --- Sidebar: user inputs ---
st.sidebar.header("Portfolio Settings")

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,GS,JPM,MSFT")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

st.sidebar.write("Weights (must sum to 1):")
weights = []
for t in tickers:
    w = st.sidebar.number_input(f"{t} weight", min_value=0.0, max_value=1.0,
                                  value=round(1/len(tickers), 2), step=0.05)
    weights.append(w)
weights = np.array(weights)

confidence_level = st.sidebar.slider("Confidence level (VaR/CVaR)", 0.90, 0.99, 0.95, 0.01)
distribution = st.sidebar.radio("Shock distribution", ["Normal", "Student-t (fat tails)"])
num_simulations = st.sidebar.slider("Number of simulations", 1000, 20000, 10000, 1000)
num_days = st.sidebar.slider("Forecast horizon (trading days)", 30, 504, 252, 21)
initial_investment = st.sidebar.number_input("Initial portfolio value ($)", value=10000, step=1000)

if abs(weights.sum() - 1.0) > 0.01:
    st.sidebar.warning(f"Weights sum to {weights.sum():.2f}, not 1.0 — results may be misleading.")

# --- Section 1: Historical data & correlation ---
st.header("1. Historical Data & Correlation")

data = yf.download(tickers, period="2y", auto_adjust=True)
log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

cov_matrix = log_returns.cov()
correlation_matrix = log_returns.corr()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Covariance Matrix")
    st.dataframe(cov_matrix)
with col2:
    st.subheader("Correlation Matrix")
    st.dataframe(correlation_matrix)

# Correlation heatmap
fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
im = ax_corr.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax_corr.set_xticks(range(len(tickers)))
ax_corr.set_yticks(range(len(tickers)))
ax_corr.set_xticklabels(tickers)
ax_corr.set_yticklabels(tickers)
for i in range(len(tickers)):
    for j in range(len(tickers)):
        ax_corr.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')
fig_corr.colorbar(im, ax=ax_corr)
ax_corr.set_title("Correlation Heatmap")
st.pyplot(fig_corr)

mean_returns = log_returns.mean()
volatilities = log_returns.std()

# --- Section 2: Monte Carlo simulation ---
st.header("2. Monte Carlo Simulation")

n_assets = len(tickers)
L = np.linalg.cholesky(cov_matrix.values)

if distribution == "Normal":
    Z = np.random.normal(size=(num_days, num_simulations, n_assets))
else:
    df = 5
    Z_raw = stats.t.rvs(df=df, size=(num_days, num_simulations, n_assets))
    scaling_factor = np.sqrt((df - 2) / df)
    Z = Z_raw * scaling_factor

correlated_Z = Z @ L.T

daily_log_returns = (mean_returns.values - 0.5 * volatilities.values**2) + correlated_Z
cumulative_log_returns = np.cumsum(daily_log_returns, axis=0)

start_prices = data['Close'].dropna().iloc[-1].values
simulated_prices = start_prices * np.exp(cumulative_log_returns)

asset_returns = simulated_prices / start_prices - 1
portfolio_returns = asset_returns @ weights
portfolio_value = initial_investment * (1 + portfolio_returns)

st.write(f"Simulated {num_simulations:,} paths over {num_days} trading days using {distribution} shocks.")

# Plot a sample of simulated portfolio paths
fig_paths, ax_paths = plt.subplots(figsize=(12, 5))
ax_paths.plot(portfolio_value[:, :200], linewidth=0.5, alpha=0.5)
ax_paths.set_title(f"{num_simulations:,} Simulated Portfolio Paths ({num_days} Trading Days)")
ax_paths.set_xlabel("Trading Days Ahead")
ax_paths.set_ylabel("Portfolio Value ($)")
ax_paths.grid(alpha=0.3)
st.pyplot(fig_paths)

# --- Section 3: VaR / CVaR ---
st.header("3. Risk Metrics")

final_day_returns = portfolio_returns[-1]
tail_pct = (1 - confidence_level) * 100
VaR = np.percentile(final_day_returns, tail_pct)
CVaR = final_day_returns[final_day_returns <= VaR].mean()

ending_values = portfolio_value[-1]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean Ending Value", f"${ending_values.mean():,.2f}")
m2.metric(f"VaR ({confidence_level:.0%})", f"{VaR:.2%}")
m3.metric(f"CVaR ({confidence_level:.0%})", f"{CVaR:.2%}")
m4.metric("Distribution", distribution)

# Histogram of ending portfolio returns
fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
ax_hist.hist(final_day_returns, bins=100, color='steelblue', alpha=0.7)
ax_hist.axvline(VaR, color='red', linestyle='--', label=f'VaR ({confidence_level:.0%})')
ax_hist.set_title("Distribution of Simulated Portfolio Returns")
ax_hist.set_xlabel("Return")
ax_hist.legend()
st.pyplot(fig_hist)