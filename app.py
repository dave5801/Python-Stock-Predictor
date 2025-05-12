from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from models import StockDataHandler, StockPredictor, StockVisualizer

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

