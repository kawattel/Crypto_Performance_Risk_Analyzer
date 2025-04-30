Kurt Wattelet -
Personal Project for Data Analytics I

# 🧮 Crypto Portfolio Performance & Risk Analyzer

This Streamlit dashboard allows users to upload historical cryptocurrency price data, define custom portfolio weights, and analyze portfolio performance and risk metrics — including Sharpe Ratio, Sortino Ratio, Max Drawdown, and Volatility.

---

## 🚀 Features

- 📤 Upload your own CSV file with historical crypto prices
- ⚖️ Set custom portfolio weights
- 📅 Select a date range for analysis
- 🏦 Adjust the risk-free rate for more realistic Sharpe/Sortino ratios
- 📈 Visualize:
  - Historical price trends
  - Rolling annualized volatility
  - Correlation heatmap between crypto assets
- 📊 Analyze:
  - Sharpe Ratio (risk-adjusted return)
  - Sortino Ratio (downside-risk adjusted return)
  - Max Drawdown (largest loss from peak)
  - Volatility (annualized standard deviation)

---

## 📁 Example CSV Format

| Date       | BTC-USD | ETH-USD |
|------------|---------|---------|
| 2023-01-01 | 16625   | 1200    |
| 2023-01-02 | 16800   | 1225    |
| ...        | ...     | ...     |

- The `Date` column should be the index.
- Column names should match the tickers you enter in the app (e.g., BTC-USD, ETH-USD).

---

## 🛠️ Setup Instructions

1. **Clone or download** the repository.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows

Install required dependencies

Launch the dashboard:
streamlit run path/to/your/dashboard_file.py


