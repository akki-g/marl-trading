import yfinance as yf
import pandas as pd


# Download full historical data for the tickers
# Adjust start date if needed

# Display the first few rows of the data
def get_data(data):
    tickers = input("Enter the tickers you want to download separated by commas: ").split(",")
    data = yf.download(tickers, start="1900-01-01")  
    return data.head()

