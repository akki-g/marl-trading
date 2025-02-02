import yfinance as yf
import pandas as pd


# Download full historical data for the tickers
# Adjust start date if needed

# Display the first few rows of the data
def get_data(tickers):
    data = yf.download(tickers, start="2000-01-01")  
    return data

