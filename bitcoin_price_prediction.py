import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

btc_data = yf.download('BTC-USD', start='2017-01-01', end='2025-01-01')

btc_data = btc_data[['Close']]

plt.figure(figsize=(10, 6))
plt.plot(btc_data['Close'])
plt.title('Bitcoin Closing Price (2017-2025)')
plt.xlabel('Date')
plt.ylabel('Price (in USD)')
plt.show()

btc_data['Prev Close'] = btc_data['Close'].shift(1)
btc_data['Moving Average (5 days)'] = btc_data['Close'].rolling(window=5).mean()
btc_data['Moving Average (20 days)'] = btc_data['Close'].rolling(window=20).mean()

btc_data = btc_data.dropna()

X = btc_data[['Prev Close', 'Moving Average (5 days)', 'Moving Average (20 days)']]
y = btc_data['Close']

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, color='blue', label='Actual Price')
plt.plot(y_test.index, y_pred, color='red', label='Predicted Price')
plt.title('Bitcoin Price Prediction (Test Data)')
plt.xlabel('Date')
plt.ylabel('Price (in USD)')
plt.legend()
plt.show()
