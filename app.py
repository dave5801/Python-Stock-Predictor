from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form["ticker"]
        df = yf.download(ticker, start="2022-01-01", end="2023-12-31")

        # Prepare features and labels
        df = df[["Close"]].copy()
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        X = df[["Close"]][:-1]
        y = df["Target"][:-1]

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index[:-1], y, label="Actual")
        plt.plot(df.index[:-1], predictions, label="Predicted")
        plt.title(f"{ticker.upper()} Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plot_path = os.path.join("static", "plot.png")
        plt.savefig(plot_path)
        plt.close()

        return render_template("index.html", plot_path=plot_path, ticker=ticker)

    return render_template("index.html", plot_path=None)

if __name__ == "__main__":
    app.run(debug=True)
