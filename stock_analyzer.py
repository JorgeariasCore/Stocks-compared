import yfinance as yf
import pandas as pd
import numpy as np


# =========================
# CONFIGURATION
# =========================
TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY"]
START_DATE = "2023-01-01"
END_DATE = "2026-01-01"


# =========================
# DOWNLOAD DATA
# =========================
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Keep only Close prices
    if "Close" in data.columns:
        close_prices = data["Close"]
    else:
        close_prices = data

    # Remove rows with all missing values
    close_prices = close_prices.dropna(how="all")

    return close_prices


# =========================
# CALCULATIONS
# =========================
def calculate_daily_returns(prices):
    return prices.pct_change().dropna() #ercentage change between consecutive values and delete enpty rows


def calculate_cumulative_returns(daily_returns):
    return (1 + daily_returns).cumprod() - 1 #umulative product It multiplies values one after another down the column


def calculate_volatility(daily_returns):
    # Annualized volatility assuming 252 trading days
    return daily_returns.std() * np.sqrt(252) #Annualized Volatility


def calculate_moving_averages(prices, short_window=20, long_window=50):
    ma_short = prices.rolling(window=short_window).mean() #moving window over your data and computes the average of values.
    ma_long = prices.rolling(window=long_window).mean()
    return ma_short, ma_long


def best_and_worst_days(daily_returns):
    summary = {}

    for ticker in daily_returns.columns:
        best_day = daily_returns[ticker].idxmax() #give me the index (position/label) where the maximum value occurs.
        worst_day = daily_returns[ticker].idxmin()

        summary[ticker] = {
            "best_day_date": best_day,
            "best_day_return": daily_returns[ticker].max(), #give the maximum value 
            "worst_day_date": worst_day,
            "worst_day_return": daily_returns[ticker].min(), #give the minimum value
        }

    return summary


# =========================
# SUMMARY TABLE
# =========================
def build_summary(prices, daily_returns, cumulative_returns):
    summary = pd.DataFrame(index=prices.columns)

    summary["Last Price"] = prices.iloc[-1]
    summary["Mean Daily Return"] = daily_returns.mean()
    summary["Volatility (Annualized)"] = daily_returns.std() * np.sqrt(252)
    summary["Total Cumulative Return"] = cumulative_returns.iloc[-1]

    return summary.sort_values(by="Total Cumulative Return", ascending=False)


# =========================
# MAIN
# =========================
def main():
    print("Downloading stock data...")
    prices = download_data(TICKERS, START_DATE, END_DATE)

    print("\nClose Prices:")
    print(prices.tail())

    daily_returns = calculate_daily_returns(prices)
    cumulative_returns = calculate_cumulative_returns(daily_returns)
    volatility = calculate_volatility(daily_returns)
    ma_20, ma_50 = calculate_moving_averages(prices)
    extremes = best_and_worst_days(daily_returns)
    summary = build_summary(prices, daily_returns, cumulative_returns)

    print("\n================ SUMMARY TABLE ================\n")
    print(summary)

    print("\n================ VOLATILITY ================\n")
    print(volatility)

    print("\n================ BEST AND WORST DAYS ================\n")
    for ticker, values in extremes.items():
        print(f"{ticker}:")
        print(f"  Best Day:  {values['best_day_date'].date()} | Return: {values['best_day_return']:.2%}")
        print(f"  Worst Day: {values['worst_day_date'].date()} | Return: {values['worst_day_return']:.2%}")
        print()

    print("\n================ LAST MOVING AVERAGES ================\n")
    for ticker in prices.columns:
        print(f"{ticker}:")
        print(f"  Last Price: {prices[ticker].iloc[-1]:.2f}")
        print(f"  20-Day MA:  {ma_20[ticker].iloc[-1]:.2f}")
        print(f"  50-Day MA:  {ma_50[ticker].iloc[-1]:.2f}")
        print()


if __name__ == "__main__":
    main()