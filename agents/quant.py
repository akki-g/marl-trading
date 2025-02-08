import numpy as np
import pandas as pd
import yfinance as yf
import torch as tor



class QuantAgent:
    def __init__(self,data):
        self.data = data
        self.slow = 50
        self.fast = 20
        self.period = 14
        self.period_long = 26
        self.period_short = 12
        self.period_signal = 9
        self.window = 60
        self.features = None
        
    def get_technical_indicators(self):
        self.data = get_SMAs(self.data, self.slow, self.fast)
        self.data = get_ema(self.data, self.period)
        self.data = get_macd(self.data, self.period_long, self.period_short, self.period_signal)
        self.data = get_rsi(self.data, self.period)
        self.data = get_BollingerBands(self.data, self.period)
        self.data = get_beta(self.data, self.window)
        self.data = set_signals(self.data)
        return self.data





def get_SMAs(data, slow, fast):
    data['SMA_Fast'] = data['Close'].rolling(window=fast).mean()
    data['SMA_Slow'] = data['Close'].rolling(window=slow).mean()
    return data


def get_ema(data, period):
    data['EMA'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

    
def get_macd(data, period_long, period_short, period_signal):
    shortEMA = data['Close'].ewm(span=period_short, adjust=False).mean()
    longEMA = data['Close'].ewm(span=period_long, adjust=False).mean()
    data['MACD'] = shortEMA - longEMA
    data['Signal_Line'] = data['MACD'].ewm(span=period_signal, adjust=False).mean()
    return data

    
def get_rsi(data, period):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))
    return data
 

def get_BollingerBands(data, period):
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['20dSTD'] = data['Close'].rolling(window=period).std()
    data['UpperBand'] = data['SMA'] + (data['20dSTD'] * 1)
    data['LowerBand'] = data['SMA'] - (data['20dSTD'] * 1)
    return data

def get_beta(data, window):
    data['Return'] = np.log(data['Close']).diff()
    benchmark = yf.download('^GSPC', start='1900-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    benchmark['BenchReturn'] = np.log(benchmark['Close']).diff()
    
    combined = pd.concat([data['Return'], benchmark['BenchReturn']], axis= 1).dropna()
    combined['Covariance'] = combined['Return'].rolling(window = window).cov(combined['BenchReturn'])
    combined['Variance'] = combined['BenchReturn'].rolling(window = window).var()

    combined['Beta'] = combined['Covariance'] / combined['Variance']
    data['Beta'] = combined['Beta']

    return data


def set_signals(data):
    data = data.dropna()  
    data = data.reset_index()
    
    # SMA signal
    data['SMAsignal'] = 0
    data.loc[data['SMA_Fast'] > data['SMA_Slow'], 'SMAsignal'] = 1
    data.loc[data['SMA_Fast'] <= data['SMA_Slow'], 'SMAsignal'] = 0

    # EMA signal
    data['EMAsignal'] = 0
    data.loc[data['Close'] > data['EMA'], 'EMAsignal'] = 1
    data.loc[data['Close'] <= data['EMA'], 'EMAsignal'] = 0

    # MACD signal
    data['MACDsignal'] = 0
    data.loc[data['MACD'] > data['Signal_Line'], 'MACDsignal'] = 1
    data.loc[data['MACD'] <= data['Signal_Line'], 'MACDsignal'] = 0

    # RSI signal
    data['RSIsignal'] = 0
    data.loc[data['RSI'] > 70, 'RSIsignal'] = 1
    data.loc[data['RSI'] < 30, 'RSIsignal'] = 0

    # Bollinger Bands signal
    data['BBsignal'] = 0
    data.loc[data['Close'] > data['UpperBand'], 'BBsignal'] = 1
    data.loc[data['Close'] < data['LowerBand'], 'BBsignal'] = 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
  
    return data