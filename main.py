from data_loader import fetch_stock_data
from preprocessing import preprocess_data, create_sequences
from model import build_lstm_model
from visualization import plot_predictions
import numpy as np
import pandas as pd

if __name__ == "__main__":
    stock = input('Enter stock ticker (e.g. AAPL): ')
    start = input('Enter start date (YYYY-MM-DD): ')
    end = input('Enter end date (YYYY-MM-DD): ')

    # Fetch data
    df = fetch_stock_data(stock, start, end)
    print('Downloaded DataFrame columns:', df.columns)
    if df.empty:
        print('Downloaded DataFrame is empty. Please check your ticker and date range.')
        exit()
    print(df.head())

    # Preprocess
    scaled_data, scaler, data = preprocess_data(df)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = create_sequences(train_data)

    # Build and train model
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=1, epochs=5)

    # Test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, _ = create_sequences(test_data)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Prepare for visualization
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plot_predictions(train, valid)
    print(valid.tail())
