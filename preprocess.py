import pandas as pd
import requests
import ta
from ta import momentum, trend
from requests.auth import HTTPBasicAuth


class DataReader(object):
    def __init__(self, symbol, at, last, period):
        self.id = '***'
        self.password = '***'
        self.symbol   = symbol
        self.at       = at
        self.last     = last
        self.period   = period
        self.raw_data = None

    def read_data(self):
        symbol_url    = f"{self.symbol}.E.BIST"
        url           = f"http://***{symbol_url}***{self.period}***{self.at}***{self.last}" # URL
        response      = requests.get(url, auth=HTTPBasicAuth(self.id, self.password))
        json_response = response.json()

        data         = pd.DataFrame(json_response)
        data.columns = ['date', 'open', 'high', 'low', 'close']
        data['date'] = pd.to_datetime(data['date'], unit='ms')
        data.set_index('date', inplace=True)

        self.raw_data = data

    def extract_indicators(self):
        data = self.raw_data.copy()

        # Calculate RSI (Relative Strength Index)
        data['RSI'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()

        # Calculate MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close=data['close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_diff'] = macd.macd_diff()

        # Calculate EMA (Exponential Moving Average)
        data['EMA'] = ta.trend.EMAIndicator(close=data['close'], window=14).ema_indicator()

        # Calculate percentage change of the close price
        data['pct_change'] = data['close'].pct_change()  # Multiply by 100 to express it as a percentage

        # Drop rows with missing values
        data.dropna(inplace=True)

        self.data = data

    def split_data(self):
        split = int(self.data.shape[0] * 0.8)

        train = self.data[:split]
        test = self.data[split:]
        return train, test
