import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions(train, valid):
    """Plot actual and predicted stock prices."""
    plt.figure(figsize=(16,8))
    plt.title('LSTM Model - Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Actual', 'Predictions'])
    plt.show()
