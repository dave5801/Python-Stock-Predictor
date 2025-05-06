from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    error = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()

        try:
            # Fetch stock data
            df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
            df = df[['Close']]
            df['Next_Close'] = df['Close'].shift(-1)
            df = df[:-1]

            X = df[['Close']]
            y = df['Next_Close']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            # Plot predictions
            plt.figure(figsize=(8, 5))
            plt.plot(y_test.values, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.title(f"{ticker} Stock Price Prediction")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join('static', 'plot.png')
            plt.savefig(plot_path)
            plt.close()
            plot_url = plot_path

        except Exception as e:
            error = f"Error fetching data for {ticker}. Check symbol."

    return render_template('index.html', plot_url=plot_url, error=error)

if __name__ == '__main__':
    app.run(debug=True)
