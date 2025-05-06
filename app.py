from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_filename = None
    ticker = None
    error = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()

        try:
            # Fetch stock data
            df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
            if df.empty:
                raise ValueError("No data returned.")

            df = df[['Close']]
            df['Next_Close'] = df['Close'].shift(-1)
            df = df.dropna()

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

            # Save plot in /static
            plot_filename = 'plot.png'
            plot_path = os.path.join('static', plot_filename)
            plt.savefig(plot_path)
            plt.close()

        except Exception as e:
            error = f"Could not retrieve or process data for {ticker}. Error: {e}"

    return render_template('index.html', plot_path=plot_filename, ticker=ticker, error=error)

if __name__ == '__main__':
    app.run(debug=True)
