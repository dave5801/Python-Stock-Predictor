import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class StockDataHandler:
    def __init__(self, ticker):
        self.ticker = ticker
        self.scaler = MinMaxScaler()
        self.sequence_length = 60

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        return yf.download(self.ticker, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))

    def prepare_data(self, data):
        """Prepare and scale the data for model training"""
        scaled_data = self.scaler.fit_transform(data[['Close']])
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i])
        return np.array(X), np.array(y)

    def split_data(self, X, y):
        """Split data into training and testing sets"""
        split = int(0.8 * len(X))
        return X[:split], y[:split], X[split:], y[split:] 