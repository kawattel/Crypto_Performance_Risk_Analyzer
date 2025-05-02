import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- App Config ---
st.set_page_config(page_title="Crypto Portfolio Analyzer", layout="wide")
st.title("Crypto Portfolio Performance & Risk Analyzer")

# --- Sidebar: Portfolio Setup ---
st.sidebar.header("Portfolio Setup")

# 📂 Data Management
st.sidebar.subheader("📂 Data Management")
sample_df = pd.DataFrame({
    "Date": ["2023-01-01", "2023-01-02"],
    "BTC-USD": [16625, 16800],
    "ETH-USD": [1200, 1225]
})
sample_df.set_index("Date", inplace=True)
csv_bytes = sample_df.to_csv().encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Sample CSV",
    data=csv_bytes,
    file_name="sample_crypto_prices.csv",
    mime="text/csv"
)
uploaded_file = st.sidebar.file_uploader("Upload Your Portfolio Data", type=["csv"])

# 🔢 Portfolio Configuration
st.sidebar.subheader("🔢 Portfolio Configuration")
tickers = st.sidebar.text_input(
    "Enter crypto tickers (comma separated):", 
    value="BTC-USD,ETH-USD"
).upper().replace(" ", "").split(",")

weights_input = st.sidebar.text_input(
    "Enter weights (comma separated, must sum to 1.0):", 
    value="0.6,0.4"
)
weights = list(map(float, weights_input.split(",")))

if len(tickers) != len(weights):
    st.error("Number of tickers and weights must match.")
    st.stop()
if not abs(sum(weights) - 1.0) < 0.01:
    st.error("Weights must sum to 1.0.")
    st.stop()

portfolio = dict(zip(tickers, weights))
st.sidebar.write("Portfolio:", portfolio)

# 🏦 Risk & Date Settings
st.sidebar.subheader("🏦 Risk & Date Settings")
risk_free_rate_input = st.sidebar.number_input(
    "Risk-Free Rate (%)", 
    value=0.0, 
    step=0.1, 
    format="%.2f"
)
risk_free_rate = risk_free_rate_input / 100

# --- Tabs ---
tab1, tab2 = st.tabs(["📈 Dashboard", "ℹ️ About This App"])

with tab1:
    if uploaded_file is not None:
        prices = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        st.success("✅ File uploaded and loaded successfully!")
    else:
        st.warning("⚠️ Please upload a CSV file to proceed.")
        st.stop()

    if prices is None or prices.empty:
        st.error("Price data not available for selected tickers.")
        st.stop()

    # --- Date Range Selector ---
    st.sidebar.subheader("Select Date Range for Analysis")
    min_date = prices.index.min().date()
    max_date = prices.index.max().date()
    start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date   = st.sidebar.date_input("End Date",   min_value=min_date, max_value=max_date, value=max_date)
    prices = prices.loc[(prices.index.date >= start_date) & (prices.index.date <= end_date)]

    # --- Return Calculations ---
    returns = prices.pct_change().dropna()
    weighted_returns = returns.copy()
    for ticker, weight in portfolio.items():
        weighted_returns[ticker] *= weight
    portfolio_returns = weighted_returns.sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # --- Beta vs BTC Calculation ---
    if "BTC-USD" in returns.columns:
        btc_returns = returns["BTC-USD"]
        aligned = pd.concat([portfolio_returns, btc_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "btc"]
        cov_matrix = np.cov(aligned["portfolio"], aligned["btc"])
        beta_btc = cov_matrix[0, 1] / cov_matrix[1, 1]
    else:
        beta_btc = np.nan

    # --- Portfolio Performance Summary ---
    starting_value = cumulative_returns.iloc[0]
    ending_value   = cumulative_returns.iloc[-1]
    cumulative_return = (ending_value / starting_value) - 1

    st.subheader("📈 Portfolio Performance Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Starting Value (Indexed)", f"{starting_value:.2f}")
    c2.metric("Ending Value (Indexed)",   f"{ending_value:.2f}")
    c3.metric("Total Cumulative Return",  f"{cumulative_return:.2%}")
    st.divider()

    # --- Metric Calculation Functions ---
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
        excess = returns - risk_free_rate / periods_per_year
        return excess.mean() / excess.std() * np.sqrt(periods_per_year)

    def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
        excess = returns - risk_free_rate / periods_per_year
        downside = excess[excess < 0]
        return excess.mean() / downside.std() * np.sqrt(periods_per_year) if downside.std() != 0 else np.nan

    def calculate_max_drawdown(cum_ret):
        rm = cum_ret.cummax()
        dd = cum_ret / rm - 1.0
        return dd.min()

    def calculate_annualized_volatility(returns, periods_per_year=252):
        return returns.std() * np.sqrt(periods_per_year)

    # --- Visualization Functions ---
    def plot_rolling_volatility(returns, window=30):
        rv = returns.rolling(window=window).std() * np.sqrt(252)
        fig, ax = plt.subplots(figsize=(10,4))
        fig.patch.set_facecolor("whitesmoke")
        ax.plot(rv, label=f"{window}-Day Rolling Volatility", color="orange")
        ax.set_title("Rolling Annualized Volatility")
        ax.set_xlabel("Date"); ax.set_ylabel("Volatility")
        ax.grid(True); ax.legend()
        st.pyplot(fig)

    def plot_correlation_heatmap(df):
        cm = df.corr()
        fig, ax = plt.subplots(figsize=(6,5))
        fig.patch.set_facecolor("whitesmoke")
        sns.heatmap(cm, annot=True, cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Between Crypto Assets")
        st.pyplot(fig)

    def plot_drawdown(cum_ret):
        rm = cum_ret.cummax()
        dd = cum_ret / rm - 1.0
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(dd, color="red", label="Drawdown")
        ax.set_title("Portfolio Drawdown Over Time")
        ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
        ax.grid(True)
        st.pyplot(fig)

    def plot_risk_return_scatter(df, port_ret):
        mr = df.mean() * 252
        vr = df.std() * np.sqrt(252)
        pm = port_ret.mean() * 252
        pv = port_ret.std() * np.sqrt(252)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(vr, mr)
        for i, t in enumerate(mr.index):
            ax.annotate(t, (vr[i], mr[i]))
        ax.scatter(pv, pm, marker="*", s=200, label="PORTFOLIO")
        ax.annotate("PORTFOLIO", (pv, pm))
        ax.set_title("Risk vs Return Scatter")
        ax.set_xlabel("Annualized Volatility"); ax.set_ylabel("Annualized Return")
        ax.grid(True); ax.legend()
        st.pyplot(fig)

    # 📈 Portfolio Price Trends
    with st.expander("📈 Portfolio Price Trends", expanded=True):
        st.line_chart(prices)
        st.caption("This chart shows the historical price trends of your selected crypto assets.")
    st.divider()

    # 📊 Portfolio Risk & Return Metrics
    with st.expander("📊 Portfolio Risk & Return Metrics", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)

        sharpe  = calculate_sharpe_ratio(portfolio_returns, risk_free_rate=risk_free_rate)
        sortino = calculate_sortino_ratio(portfolio_returns, risk_free_rate=risk_free_rate)
        max_dd  = calculate_max_drawdown(cumulative_returns)
        vol     = calculate_annualized_volatility(portfolio_returns)

        # Sharpe
        sc = "green" if sharpe>1 else "orange" if sharpe>0 else "red"
        col1.markdown(f"<h3 style='color:{sc};'>{sharpe:.2f}</h3>", unsafe_allow_html=True)
        col1.caption("Sharpe Ratio")

        # Sortino
        soc = "green" if sortino>1 else "orange" if sortino>0 else "red"
        col2.markdown(f"<h3 style='color:{soc};'>{sortino:.2f}</h3>", unsafe_allow_html=True)
        col2.caption("Sortino Ratio")

        # Max Drawdown
        dc = "green" if max_dd>-0.2 else "orange" if max_dd>-0.5 else "red"
        col3.markdown(f"<h3 style='color:{dc};'>{max_dd:.2%}</h3>", unsafe_allow_html=True)
        col3.caption("Max Drawdown")

        # Volatility
        vc = "green" if vol<0.30 else "orange" if vol<0.60 else "red"
        col4.markdown(f"<h3 style='color:{vc};'>{vol:.2%}</h3>", unsafe_allow_html=True)
        col4.caption("Annualized Volatility")

        # Beta vs BTC
        col5.metric("Beta vs BTC", f"{beta_btc:.2f}" if not np.isnan(beta_btc) else "N/A")

        st.subheader("Portfolio Weights")
        pdf = pd.DataFrame.from_dict(portfolio, orient='index', columns=['Weight'])
        st.table(pdf)
        st.caption("This table shows each asset's weight in your custom portfolio.")

        # --- Metrics CSV Export with Beta vs BTC
        metrics_data = {
            "Sharpe Ratio": [sharpe],
            "Sortino Ratio": [sortino],
            "Max Drawdown": [max_dd],
            "Annualized Volatility": [vol],
            "Cumulative Return": [cumulative_return],
            "Beta vs BTC": [round(beta_btc,4) if not np.isnan(beta_btc) else "N/A"],
            "Start Date": [start_date],
            "End Date": [end_date],
            "Risk-Free Rate (%)": [risk_free_rate_input],
            "Portfolio Weights": [str(portfolio)]
        }
        mdf = pd.DataFrame(metrics_data)
        csv_out = mdf.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Portfolio Metrics as CSV",
            data=csv_out,
            file_name="portfolio_metrics.csv",
            mime="text/csv"
        )

        st.success("✅ Portfolio risk and return metrics calculated successfully!")
    st.divider()

    # 📉 Rolling Volatility Over Time
    with st.expander("📉 Rolling Volatility Over Time"):
        plot_rolling_volatility(portfolio_returns)
        st.caption("This plot shows how your portfolio's volatility evolves over time using a rolling 30-day window.")
    st.divider()

    # 📉 Drawdown Over Time
    with st.expander("📉 Drawdown Over Time"):
        plot_drawdown(cumulative_returns)
        st.caption("Shows peak-to-trough drawdown of your portfolio.")
    st.divider()

    # 🔥 Correlation Between Crypto Assets
    with st.expander("🔥 Correlation Between Crypto Assets"):
        plot_correlation_heatmap(returns)
        st.caption("This heatmap shows the correlation between different crypto assets based on their daily returns.")
    st.divider()

    # 🎯 Risk-Return Scatter Plot
    with st.expander("🎯 Risk-Return Scatter Plot"):
        plot_risk_return_scatter(returns, portfolio_returns)
        st.caption("Scatter of annualized return vs volatility for each asset (★ is portfolio).")

with tab2:
    st.header("📋 About This Crypto Portfolio Analyzer")
    st.markdown("""
    Welcome to the **Crypto Portfolio Performance & Risk Analyzer**! 🚀

    ---
    ### 📈 What This Dashboard Allows You To Do:
    - **Upload historical crypto price data** (via CSV)
    - **Assign custom portfolio weights** to each crypto
    - **Select custom date ranges** for focused analysis
    - **Adjust the assumed risk-free rate** to simulate different economic conditions
    - **Visualize** your portfolio's **price trends** over time
    - **Analyze key performance and risk metrics**, including:
        - **Sharpe Ratio**: Measures risk-adjusted return. Higher is better.
        - **Sortino Ratio**: Focuses only on downside volatility (bad volatility).
        - **Max Drawdown**: Largest peak-to-trough loss during the period.
        - **Volatility**: Annualized standard deviation of returns (higher = more risk).

    ---
    ### 🏦 Understanding the Risk-Free Rate:
    - The **risk-free rate** represents the return investors could earn from an absolutely safe investment, such as U.S. Treasury bills.
    - In this dashboard, the risk-free rate is used in calculating the **Sharpe Ratio** and **Sortino Ratio**.
    - Allowing users to set a custom risk-free rate makes the analysis more realistic by accounting for different economic environments.
    - The risk-free rate adjusts the Sharpe and Sortino Ratios to show how much extra return was earned above a safe investment, making risk-adjusted performance more realistic.

    **Example:**  
    > If a portfolio earns 10% annual return with 20% volatility, and the risk-free rate is 0%, the Sharpe Ratio would be **0.50**.  
    > However, if the risk-free rate rises to 3%, the Sharpe Ratio drops to **0.35** — reflecting that only 7% of the return was truly earned above a safe investment.

    ---
    ### ⚡ Ideal Use Cases:
    - Quickly evaluate crypto portfolio **risk/return** characteristics
    - Experiment with **different weighting strategies**
    - Analyze how **portfolio risk evolves** across time
    - Build intuition about how **individual assets contribute** to performance

    ---
    """)
