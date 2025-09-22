import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """Preprocess and scale closing price data."""
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, scaler, data

def create_sequences(scaled_data, seq_length=60):
    """Create sequences for LSTM input."""
    x, y = [], []
    for i in range(seq_length, len(scaled_data)):
        x.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y
