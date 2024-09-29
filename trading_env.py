import numpy as np

class TradingSystem_v0:
    def __init__(self, data, k_value):
        self.data          = data
        self.k             = k_value
        self.total_steps   = len(self.data) - self.k
        self.current_step  = 0
        self.initial_state = np.array(self.data[:self.k]).flatten()
        self.state         = self.initial_state
        self.reward        = 0.0
        self.is_terminal   = False
        self.position      = None
        self.buy_price     = None

    def step(self, action):
        self.current_step += 1
        if self.current_step == self.total_steps:
            self.is_terminal = True
        self.reward = (action - 1) * self.data['pct_change'].iloc[self.current_step + self.k - 1]
        self.state = np.array(self.data.iloc[self.current_step:(self.k + self.current_step)]).flatten()
        return self.state, self.reward, self.is_terminal

    def reset(self):
        self.total_steps = len(self.data) - self.k
        self.current_step = 0
        self.initial_state = np.array(self.data[:self.k]).flatten()
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False
        self.position = None
        self.buy_price = None

        return self.state