"""
Neural Network Strategy - TSLA stocks

@author: Kristijan <kristijan.sarin@gmail.com>
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import datetime

# Fetch interval data
def fetch_data(ticker):
    data = pd.DataFrame()
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    while start_date < end_date:
        temp_data = yf.download(ticker, interval='1m', start=start_date, end=start_date + datetime.timedelta(days=1))
        data = pd.concat([data, temp_data])
        start_date += datetime.timedelta(days=1)
    return data

# Fetch TESLA data
ticker = 'TSLA'
data = fetch_data(ticker)

# Preprocess data
data['Return'] = data['Close'].pct_change()
data['Direction'] = np.where(data['Return'] > 0, 1, 0)

# Feature engineering
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data.dropna(inplace=True)

# Features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']
X = data[features].shift().dropna()
y = data['Direction'][1:]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Training the model
model = MLPClassifier(hidden_layer_sizes=(100, 100, 50), max_iter=1000, random_state=42, learning_rate='adaptive', activation='relu')
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Backtest
initial_balance = 10000
balance = initial_balance
side_bank = 0
positions = []
investment = 1000  # initial investment amount

for i in range(len(predictions)):
    if predictions[i] == 1:  # Buy signal
        balance *= (1 + data['Return'].iloc[i + len(y_train)])
        investment = 1000  # reset investment after a gain
    else:  # Sell signal
        if balance * 0.5 <= investment:  # Check if balance can cover the loss
            balance *= 0.5
            side_bank += investment  # Add to side bank if balance is not enough
        else:
            balance -= investment
        investment *= 2
    positions.append(balance)

# Visuals
plt.figure(figsize=(14, 7))
plt.plot(range(len(positions)), positions, label='Portfolio Value')
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.title('Backtest of Neural Networks Trading Strategy on TESLA')
plt.legend()
plt.show()

# Display
final_balance = positions[-1]
print(f'Initial Balance: ${initial_balance}')
print(f'Final Balance: ${final_balance}')
print(f'Net Profit: ${final_balance - initial_balance}')
print(f'Total Added from Side Bank: ${side_bank}')
