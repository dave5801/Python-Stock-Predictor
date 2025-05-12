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

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        try:
            df = yf.download(ticker, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))
            data = df[['Close']]

            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            # Create sequences
            sequence_length = 60
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i])
            X, y = np.array(X), np.array(y)

            # Split
            split = int(0.8 * len(X))
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

            # Predict
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Plot results
            plt.figure(figsize=(10, 5))
            dates = df.index[-len(actual):]  # Get the dates corresponding to test data
            plt.plot(dates, actual, label='Actual')
            plt.plot(dates, predictions, label='Predicted')
            plt.title(f'{ticker} Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.xticks(rotation=45)  # Rotate date labels for better readability
            plt.tight_layout()

            plot_path = os.path.join('static', 'prediction.png')
            plt.savefig(plot_path)
            plt.close()

            return render_template('index.html', ticker=ticker, image=plot_path)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

