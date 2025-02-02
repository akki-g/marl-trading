from api.data import *
import pandas as pd
import numpy as np



class TransitionMatrix:
    def __init__(self):
        """
        states: list of discrete market regimes (e.g., ["Bullish", "Neutral", "Bearish"])
        actions: list of trading actions (e.g., ["Buy", "Hold", "Sell"])
        vol_levels: list of volatility levels (e.g., ["Low", "Medium", "High"])
        """
        self.states = []
        self.transition_counts = pd.DataFrame()

    def fetch_data(self):
        data = get_data("NVDA")
        data['Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Return'].rolling(21).std()
        return data.dropna()

    def define_states(self, data):

        return_bins = [-np.inf, -0.01, 0.01, np.inf]
        return_labels = ["Bearish", "Neutral", "Bullish"]
        data['State'] = pd.cut(data['Return'], bins=return_bins, labels=return_labels)

        vol_bins = [-np.inf, 0.01, 0.02, np.inf]
        vol_labels = ["Low", "Medium", "High"]
        data['Vol_State'] = pd.cut(data['Volatility'], bins=vol_bins, labels=vol_labels)

        data['Composite_State'] = data['State'].astype(str) + " " + data['Vol_State'].astype(str)

        self.states = data['Composite_State'].unique()
        return data

    def create_transition_matrix(self, data):
        self.transition_counts = pd.DataFrame(
            np.zeros((len(self.states), len(self.states))),
            index=self.states,
            columns=self.states
        )
        for i in range(len(data) - 1):
            current = data.iloc[i]['Composite_State']
            next_state = data.iloc[i + 1]['Composite_State']
            self.transition_counts.loc[current, next_state] += 1
        
        return self.normalize_transition_matrix()

    def normalize_transition_matrix(self):

        row_sums = self.transition_counts.sum(axis=1)
        return self.transition_counts.div(row_sums, axis=0).fillna(0)


if __name__ == "__main__":
    tm = TransitionMatrix()
    data = tm.fetch_data()
    data = tm.define_states(data)
    tm.create_transition_matrix(data)
    print("Transition Matrix:")
    print(tm.transition_counts)


