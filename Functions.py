import yfinance as yff
import pandas as pd
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def get_indicator_table(ticker: str) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    df = yff.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    df.reset_index(inplace=True)

    close_prices = df['Close'].squeeze()

    df['sma_20'] = close_prices.rolling(window=20).mean()
    df['rsi_14'] = RSIIndicator(close=close_prices, window=14).rsi()
    df['bb_percent'] = BollingerBands(close=close_prices, window=20, window_dev=2).bollinger_pband()
    df['Close_Open'] = df['Close'] - df['Open']

    future_close = df['Close'].shift(-3).squeeze()
    open_price = df['Open'].squeeze()

    df['future_close'] = future_close
    df['profitable'] = (future_close.values > open_price.values).astype(int)

    # Drop NaNs caused by indicators or shifting
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[['Date', 'sma_20', 'rsi_14', 'bb_percent', 'Close_Open', 'profitable']]

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import xgboost as xgb


# Step 1: Get data
df = get_indicator_table("DUK")

# Step 2: Define features and target
X = df[['sma_20', 'rsi_14', 'bb_percent' , 'Close_Open']]
y = df['profitable']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 6: Feature Importance
xgb.plot_importance(model)
plt.tight_layout()
plt.show()

model.save_model("XGB_DUK.json")