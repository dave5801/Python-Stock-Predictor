from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
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

class StockPredictor:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape):
        """Build and compile the LSTM model"""
        self.model = Sequential([
            LSTM(50, return_sequences=False, input_shape=input_shape),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)

class StockVisualizer:
    def __init__(self, ticker):
        self.ticker = ticker

    def plot_predictions(self, dates, actual, predictions, scaler):
        """Create and save the prediction plot"""
        plt.figure(figsize=(10, 5))
        actual = scaler.inverse_transform(actual.reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions)
        
        plt.plot(dates, actual, label='Actual')
        plt.plot(dates, predictions, label='Predicted')
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = os.path.join('static', 'prediction.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        try:
            # Initialize components
            data_handler = StockDataHandler(ticker)
            predictor = StockPredictor()
            visualizer = StockVisualizer(ticker)

            # Fetch and prepare data
            df = data_handler.fetch_data()
            X, y = data_handler.prepare_data(df)
            X_train, y_train, X_test, y_test = data_handler.split_data(X, y)

            # Build and train model
            predictor.build_model((X_train.shape[1], 1))
            predictor.train(X_train, y_train)

            # Make predictions
            predictions = predictor.predict(X_test)

            # Visualize results
            dates = df.index[-len(y_test):]
            plot_path = visualizer.plot_predictions(dates, y_test, predictions, data_handler.scaler)

            return render_template('index.html', ticker=ticker, image=plot_path)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

