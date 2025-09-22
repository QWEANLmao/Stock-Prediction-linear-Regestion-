import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start, end):
    """Fetch historical stock data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError('No data found for the given ticker and date range.')
    return df
