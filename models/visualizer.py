import matplotlib.pyplot as plt
import os

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