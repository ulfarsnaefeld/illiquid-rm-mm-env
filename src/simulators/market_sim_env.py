import gymnasium as gym
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from itertools import product


class MarketSimEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, s0, T, v, q0, c0, sigma, drift, gamma, itp, ms, tf, itfp=1, seed=None):
        '''
        Description:
            A market making simulator environment.

        Action space:
            Spread, Spread Skewness

        Observation space:
            Current Price, Inventory, Time, RSI,

        Parameters:
            s0 (float):             Initial stock price
            T (float):              Episode duration
            v (float):              Verbose
            q0 (float):             Initial inventory
            c0 (float):             Initial cash on hand
            sigma (float):          Price volatility
            drift (float):          Price drift
            gamma (float):          Discount factor
            itp (float):            Percentage of Informed traders
            itfp (floa):            Informed traders future sight distance prices[current_time + itfp]
            os (float):             Maximum spread
            tf (float):             Trading frequency by investors
        '''
        super().__init__()
        self.s0 = s0
        self.episode_duration = T
        self.verbose = v

        self.initial_inventory = q0
        self.initial_cash = c0

        self.sigma = sigma
        self.drift = drift
        self.gamma = gamma

        self.itp = itp
        self.itfp = itfp
        self.ms = ms
        self.tf = tf

        # Observation space Price, Inventory, Time, RSI
        self.observation_space = gym.spaces.Box(
            low=  np.array([0.0,      -math.inf,  0.0, 0.0]),
            high= np.array([math.inf,  math.inf,  T, 100.0]),
            dtype=np.float32
            )

        # Action space
        # Define the action space 0-5% Spread and skewness from -.8 to .8
        # self.action_space = gym.spaces.Box(
        #     low=  np.array([0.0, -0.8]),
        #     high= np.array([5.0,  0.8]),
        #     dtype=np.float32
        #     )

        spread_incrementor = 0.005

        self.spread_values = np.arange(0, self.ms + spread_incrementor, spread_incrementor)

        self.skewness_values = [-0.8, -0.4, 0, 0.4, 0.8]
        self.actions = list(product(self.spread_values, self.skewness_values))

        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.rsi = 50
        self.render_df: pd.DataFrame = pd.DataFrame(columns=["Bid", "Price", "Ask", "Spread", "Skew", "Cash", "Inventory", "PnL", "RSI", "Trades", "Informed Trades", "Noise Trades", "Trade Imbalance"])

        self.reset()

    def reset(self, seed=None, option=None):
        super().reset(seed=seed)
        self.minute = 0
        self.inventory = self.initial_inventory
        self.cash = self.initial_cash
        self.pnl = 0
        self.current_price = self.s0
        self.last_portfolio_value = self.inventory * self.current_price + self.cash

        # Stats
        self.total_buys = 0
        self.total_sells = 0
        self.total_noise = 0
        self.total_informed = 0
        self.volume_imbalance = 0

        self.price_path = []
        self._initiate_price_path()
        return self._get_observation(), {}


    def step(self, action):
        # Update the current price
        self.current_price = self.price_path[self.minute]

        # Extract the spread and skewness from action
        self.spread, self.skewness = self.actions[action]

        # Set the bid and ask around the real price
        self.bid_price = self.current_price - (self.spread / 2.0) * (1 - self.skewness) * self.current_price
        self.ask_price = self.current_price + (self.spread / 2.0) * (1 + self.skewness) * self.current_price

        # Simulate market reaction to the bid and ask prices
        self._simulate_market(self.bid_price, self.ask_price)

        self.minute += 1
        reward = self._calculate_reward()
        done = self.minute >= self.episode_duration

        return self._get_observation(), reward, done, False,{}

    def _get_observation(self):
        rsi = self._calculate_rsi(self.price_path[:self.minute + 1]) if self.minute >= 14 else 50
        return np.array([self.current_price, self.inventory, self.minute, rsi], dtype=np.float32)

    def _initiate_price_path(self):
        dt = 1 / self.episode_duration
        self.price_path = [self.s0]
        for _ in range(1, self.episode_duration):
            random_shock = np.random.normal(0, self.sigma * np.sqrt(dt))
            price_change = self.drift * dt + random_shock
            new_price = self.price_path[-1] * np.exp(price_change)
            self.price_path.append(new_price)

    def _simulate_market(self, bid_price, ask_price):
        # Determine the type of trader (informed or uninformed) using Poisson process
        trade_occurrence = np.random.poisson(self.tf)

        for _ in range(trade_occurrence):
            amount = np.random.randint(1, 10)

            if np.random.rand() < self.itp:
                # Informed trader makes a trade
                self._execute_informed_trade(bid_price, ask_price, amount)
            else:
                # Uninformed trader makes a trade
                self._execute_noise_trade(bid_price, ask_price, amount)

    def _execute_informed_trade(self, bid_price, ask_price, amount):
        next_true_value = self.price_path[self.minute + self.itfp] if self.minute + self.itfp < self.episode_duration else self.current_price
        if next_true_value > ask_price: # Buy Informed Trade
            self.inventory -= amount
            self.cash += amount * ask_price
            # Stats
            self.total_buys += 1
            self.total_informed += 1
            self.volume_imbalance += amount
        elif next_true_value < bid_price: # Sell Informed Trade
            self.inventory += amount
            self.cash -= amount * bid_price
            # Stats
            self.total_sells += 1
            self.total_informed += 1
            self.volume_imbalance -= amount

    def _execute_noise_trade(self, bid_price, ask_price, amount):
        if np.random.rand() < 0.5: # Buy Noise Trade
            self.inventory -= amount
            self.cash += amount * ask_price
            # Stats
            self.total_buys += 1
            self.total_noise += 1
            self.volume_imbalance += amount
        else:
            self.inventory += amount # Sell Noise Trade
            self.cash -= amount * bid_price
            # Stats
            self.total_sells += 1
            self.total_noise += 1
            self.volume_imbalance -= amount

    def _calculate_reward(self):
        # PnL Change
        current_portfolio_value = self.inventory * self.current_price + self.cash
        pnLReward = current_portfolio_value - self.last_portfolio_value
        self.pnl += pnLReward
        self.last_portfolio_value = current_portfolio_value

        # Penalties
        inventory_penalty = 0.1 * np.abs(self.inventory - self.initial_inventory)

        return pnLReward - inventory_penalty

    def _calculate_rsi(self, prices, period=14):
        if len(prices) < period:
            self.rsi = 50
            return 50.0  # Return a neutral value if there's not enough data
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]

            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period

            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)

        self.rsi = rsi[-1]
        return rsi[-1]

    def render(self, folder_path=''):
        render_data = {
            "Bid": self.bid_price,
            "Price": self.current_price,
            "Ask": self.ask_price,
            "Spread": self.spread,
            "Skew": self.skewness,
            "Cash": self.cash,
            "Inventory": self.inventory,
            "PnL": self.pnl,
            "RSI": self.rsi,
            "Trades": self.total_buys + self.total_sells,
            "Informed Trades": self.total_informed,
            "Noise Trades": self.total_noise,
            "Trade Imbalance": self.volume_imbalance
        }
        print(render_data)

        new_data_df = pd.DataFrame([render_data])
        self.render_df = pd.concat([self.render_df, new_data_df], ignore_index=True)

        # Save to CSV
        self.render_df.to_csv(folder_path+"market_sim_render.csv", index=False)

    def stats(self):
        print(f"Total: {self.total_buys + self.total_sells}, Buys: {self.total_buys}, Sells: {self.total_sells}, Noise: {self.total_noise}, Informed: {self.total_informed}")

if __name__ == "__main__":
    env = MarketSimEnv(
            s0=100,
            T=200,
            v=True,
            q0=0,
            c0=100_000,
            sigma=0.2,
            drift=0.01,
            gamma=.99,
            itp=0.2,
            itfp=2,
            ms=0.03,
            tf=5)
    check_env(env)

    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=200_000)

    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    env.stats()
