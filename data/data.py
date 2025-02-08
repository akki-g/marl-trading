# Step 1: Install required packages
# Run this in your terminal first:
# pip install yfinance pandas numpy

import pandas as pd
import yfinance as yf
import numpy as np

class DataPipeline:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        
    def load_live_data(self, ticker='AAPL', period='5y'):
        """Load data from Yahoo Finance"""
        try:
            self.raw_data = yf.download(ticker, period=period, interval='1d')
            print(f"Successfully loaded data for {ticker}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    
    def add_technical_indicators(self):
        """Add technical indicators to the data"""
        if self.raw_data is None:
            print("No data loaded! Create or load data first")
            return
            
        df = self.raw_data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD Calculation
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['20_SMA'] = df['Close'].rolling(20).mean()
        df['20_STD'] = df['Close'].rolling(20).std()
        df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
        df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)
        
        self.processed_data = df.dropna()
        print("Successfully added technical indicators")

# ------------------------------------------------------------
# Implementation Test
# ------------------------------------------------------------
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Try loading live data
    try:
        pipeline.load_live_data()
    except:
        print("Live data loading failed, creating sample data...")
    
    # Add technical indicators
    pipeline.add_technical_indicators()
    
    # Show processed data
    print("\nProcessed Data Preview:")
    print(pipeline.processed_data[['Close', 'SMA_20', 'RSI', 'MACD']].head())
