from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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