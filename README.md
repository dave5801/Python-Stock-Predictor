# Stock Price Predictor

A Flask-based web application that predicts stock prices using Linear Regression. Enter any valid stock ticker symbol to get predictions and visualize the results.

## 🚀 Features

- Enter any valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)
- Fetches historical stock data (2020–2024)
- Predicts next-day price using Linear Regression
- Displays actual vs predicted prices in a line chart
- Clean UI with basic CSS styling

## 📷 Demo

[Add your demo screenshot here]

## 🛠️ Technologies Used

- Python 3
- Flask
- yfinance
- pandas
- scikit-learn
- matplotlib
- HTML/CSS

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/stock-price-predictor-flask.git
cd stock-price-predictor-flask
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt, install manually:

```bash
pip install flask yfinance pandas scikit-learn matplotlib
```

## 🚀 Running the App

```bash
python app.py
```

Then visit http://localhost:5000 in your browser.

## 📁 Project Structure

```
stock-price-predictor-flask/
│
├── app.py                 # Main Flask app
├── static/
│   ├── style.css          # CSS styling
│   └── plot.png           # Generated plot (overwritten each time)
├── templates/
│   └── index.html         # HTML template
└── README.md
```

## ⚠️ Notes

- This app uses basic linear regression, which is not ideal for real stock market prediction — it's educational only.
- It uses a fixed date range (2020–2024); you can easily modify this in app.py.

## ✨ Future Improvements

- Add multiple machine learning models (e.g., LSTM)
- Predict prices for multiple future days
- Make the plot interactive with Plotly
- Deploy on Render, Heroku, or Railway

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.