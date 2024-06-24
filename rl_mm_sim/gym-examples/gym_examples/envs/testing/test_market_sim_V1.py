import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

class Order:
    def __init__(self, price, amount):
        self.price = price
        self.amount = amount

class MarketSimEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 reward_function=None,
                 mm_portfolio_initial_inventory=100_000,
                 mm_portfolio_initial_capital=20_000_000,
                 max_episode_duration=240,
                 verbose=1,
                 render_mode=None):
        super().__init__()
        self.max_episode_duration = max_episode_duration
        self.verbose = verbose

        self.initial_position = mm_portfolio_initial_inventory
        self.initial_capital = mm_portfolio_initial_capital

        self.inventory = self.initial_position
        self.capital = self.initial_capital

        self.reward_function = reward_function if reward_function else self.basic_reward_function

        self.render_mode = render_mode

        # Define the discrete action space
        self.n_actions = 3  # 3 actions for each bid/ask price: decrease, no change, increase
        self.n_levels = 5  # 5 bids and 5 asks
        self.action_space = spaces.Discrete(self.n_actions ** (2 * self.n_levels))

        # Observation space: top 5 bids, top 5 asks, inventory, capital
        self.observation_space = spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max, shape=(12,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.inventory = self.initial_position
        self.capital = self.initial_capital

        self.bids = [
            Order(price, (600 * (self.n_levels - index)))
            for index, price in enumerate(np.round(np.sort(np.random.uniform(140, 150, size=self.n_levels))[::-1], 1))
        ]
        self.asks = [
            Order(price, (600 * (self.n_levels - index)))
            for index, price in enumerate(np.round(np.sort(np.random.uniform(150, 160, size=self.n_levels)), 1))
        ]

        self.true_price = np.mean([order.price for order in self.bids + self.asks])

        return self._get_observation(), {}

    def step(self, action):
        action = self._decode_action(action)
        self._apply_action(action)
        self._simulate_market()
        self.day += 1

        observation = self._get_observation()
        reward = self.reward_function()
        done = self.day >= self.max_episode_duration
        info = {}

        return observation, reward, done, False, info

    def _decode_action(self, action):
        # Decode the action into individual actions for bids and asks
        actions = []
        for _ in range(2 * self.n_levels):
            actions.append(action % self.n_actions)
            action //= self.n_actions
        return actions[::-1]

    def _apply_action(self, action):
        action_mappings = [-1, 0, 1]  # Mapping of discrete actions to continuous changes

        # while self.asks[0].amount == 0:
        #     self.asks.pop(0)
        #     self.asks.append(Order(self.asks[-1].price + 0.01, 10))

        # while self.bids[0].amount == 0:
        #     self.bids.pop(0)
        #     self.bids.append(Order(self.bids[-1].price - 0.01, 10))

        for i in range(self.n_levels):
            self.bids[i].price = np.clip(self.bids[i].price + action_mappings[action[i]], 0, np.inf)
            self.asks[i].price = np.clip(self.asks[i].price + action_mappings[action[self.n_levels + i]], 0, np.inf)

        self.bids = sorted(self.bids, key=lambda x: x.price, reverse=True)
        self.asks = sorted(self.asks, key=lambda x: x.price)

    def _simulate_market(self):
        num_market_orders = 5

        for _ in range(num_market_orders):
            if np.random.rand() < 0.5:
                if self.asks:
                    best_ask = next((order for order in self.asks if order.amount > 0), None)
                    trade_amount = min(best_ask.amount, np.random.randint(1000, 5000))  # random or or available amount
                    self.inventory -= trade_amount
                    self.capital += trade_amount * best_ask.price
                    best_ask.amount -= trade_amount
            else:
                if self.bids:
                    best_bid = next((order for order in self.bids if order.amount > 0), None)
                    trade_amount = min(best_bid.amount, np.random.randint(1000, 5000))
                    self.inventory += trade_amount
                    self.capital -= trade_amount * best_bid.price
                    best_bid.amount -= trade_amount

    def _get_observation(self):
        bids_prices = [order.price for order in self.bids]

        asks_prices = [order.price for order in self.asks]
        observation = np.concatenate([bids_prices, asks_prices, [self.inventory, self.capital]])
        return observation.astype(np.float32)

    def render(self, mode='human'):
        print(f"Day: {self.day}, Bids: {[order.price for order in self.bids]}, Asks: {[order.price for order in self.asks]}, Inventory: {self.inventory}, Capital: {self.capital}")

    def close(self):
        pass

    def basic_reward_function(self):
        # pnl = self.capital - self.initial_capital
        inventory_penalty = 0.1 * np.abs(self.inventory - self.initial_position)

        spread_penalty = 0
        for bid, ask in zip(self.bids, self.asks):
            spread = ask.price - bid.price
            if spread / bid.price > 0.02:
                spread_penalty += 0.01 * spread

        zero_amount_penalty = sum(1 for order in self.bids + self.asks if order.amount == 0)

        return 1 - inventory_penalty - spread_penalty - zero_amount_penalty

# Register the custom environment
gym.register(
    id='MarketMaking-v1',
    entry_point=MarketSimEnv,
)

if __name__ == "__main__":
    env = gym.make('MarketMaking-v1')
    check_env(env)

    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()