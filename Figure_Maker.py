import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

ticker = "AAPL"
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# Disable auto_adjust to avoid multi-index columns (for now)
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

df.reset_index(inplace=True)

# Make sure Close is a 1D Series
close_prices = df['Close'].squeeze()

# Add 20-day SMA
df['sma_20'] = close_prices.rolling(window=20).mean()

# Add RSI
rsi = RSIIndicator(close=close_prices, window=14)
df['rsi_14'] = rsi.rsi()

# Add Bollinger %B
bb = BollingerBands(close=close_prices, window=20, window_dev=2)
df['bb_percent'] = bb.bollinger_pband()

df['Close_Open'] = df['Close'] - df['Open']
# Ensure both are Series (not DataFrames)
future_close = df['Close'].shift(0).squeeze()
open_price = df['Open'].squeeze()

# Assign to DataFrame
df['future_close'] = future_close
df['profitable'] = future_close > open_price

print(df[['Date', 'Close_Open', 'sma_20', 'rsi_14', 'bb_percent']])

import matplotlib.pyplot as plt
import seaborn as sns

# Add a label for profit/loss: 1 if profitable, 0 otherwise


# Define color palette
palette = {True: 'green', False: 'red'}

# Plot settings
sns.set(style="whitegrid", context="talk")

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. RSI vs SMA
sns.scatterplot(
    x='rsi_14', y='sma_20', hue='profitable', palette=palette,
    data=df, ax=axes[0], alpha=0.6
)
axes[0].set_title('RSI vs SMA')

# 2. RSI vs Bollinger %B
sns.scatterplot(
    x='rsi_14', y='bb_percent', hue='profitable', palette=palette,
    data=df, ax=axes[1], alpha=0.6
)
axes[1].set_title('RSI vs Bollinger %B')

# 3. Bollinger %B vs SMA
sns.scatterplot(
    x='bb_percent', y='sma_20', hue='profitable', palette=palette,
    data=df, ax=axes[2], alpha=0.6
)
axes[2].set_title('Bollinger %B vs SMA')



plt.tight_layout()
plt.show()

model.save_model("XGB_AAPL.json")




