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

        # Define the action space
        self.n_levels = 5  # 5 bids and 5 asks
        self.price_bins = 21  # For price changes from -10 to +10 (in steps of 0.01)
        self.amount_bins = 21  # For amount changes from -1000 to +1000 (in steps of 10)

        # Total number of discrete actions
        self.n_actions = 3 * self.n_levels * 2 * self.price_bins * self.amount_bins
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space: top 5 bids, top 5 asks, inventory, capital
        self.observation_space = spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max, shape=(12,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.inventory = self.initial_position
        self.capital = self.initial_capital

        self.bids = [
            Order(price, (600 * (2 + index)))
            for index, price in enumerate(np.round(np.sort(np.random.uniform(140, 150, size=self.n_levels))[::-1], 2))
        ]
        self.asks = [
            Order(price, (600 * (2 + index)))
            for index, price in enumerate(np.round(np.sort(np.random.uniform(150, 160, size=self.n_levels)), 2))
        ]

        self.true_price = np.mean([order.price for order in self.bids + self.asks])

        return self._get_observation(), {}

    def step(self, action):
        action_type, level, side, price_adjustment, amount_adjustment = self._decode_action(action)
        self._apply_action(action_type, level, side, price_adjustment, amount_adjustment)
        self._simulate_market()
        self.day += 1

        observation = self._get_observation()
        reward = self.reward_function()
        done = self.day >= self.max_episode_duration
        info = {}

        return observation, reward, done, False, info

    def _encode_action(self, action_type, level, side, price_adjustment, amount_adjustment):
        action = action_type
        action = action * self.n_levels + level
        action = action * 2 + side
        action = action * self.price_bins + price_adjustment
        action = action * self.amount_bins + amount_adjustment
        return action

    def _decode_action(self, action):
        amount_adjustment = action % self.amount_bins
        action //= self.amount_bins
        price_adjustment = action % self.price_bins
        action //= self.price_bins
        side = action % 2
        action //= 2
        level = action % self.n_levels
        action //= self.n_levels
        action_type = action
        return action_type, level, side, price_adjustment, amount_adjustment

    def _sort_lob(self):
        self.bids = sorted(self.bids, key=lambda x: x.price, reverse=True)[:self.n_levels]
        self.asks = sorted(self.asks, key=lambda x: x.price)[:self.n_levels]

    def _apply_action(self, action_type, level, side, price_adjustment, amount_adjustment):
        price_adjustment = (price_adjustment - (self.price_bins // 2)) * 0.1  # Scale to a reasonable price range
        amount_adjustment = (amount_adjustment - (self.amount_bins // 2)) * 100  # Scale to a reasonable amount range

        if side == 0:  # Bid side
            if action_type == 0:  # Place a new bid
                new_price = np.round(140 + price_adjustment, 2)  # Adjust based on the price adjustment
                new_amount = max(0, 600 + amount_adjustment)  # Adjust based on the amount adjustment
                self.bids.append(Order(new_price, new_amount))
            elif action_type == 1:  # Modify an existing bid
                self.bids[level].price = np.clip(np.round(self.bids[level].price + price_adjustment, 2), 0, np.inf)
                self.bids[level].amount = np.clip(self.bids[level].amount + amount_adjustment, 0, np.inf)
            elif action_type == 2:  # Remove an existing bid
                self.bids[level].amount = 0
        else:  # Ask side
            if action_type == 0:  # Place a new ask
                new_price = np.round(150 + price_adjustment, 2)  # Adjust based on the price adjustment
                new_amount = max(0, 500 + amount_adjustment)  # Adjust based on the amount adjustment
                self.asks.append(Order(new_price, new_amount))
            elif action_type == 1:  # Modify an existing ask
                self.asks[level].price = np.clip(np.round(self.asks[level].price + price_adjustment, 2), 0, np.inf)
                self.asks[level].amount = np.clip(self.asks[level].amount + amount_adjustment, 0, np.inf)
            elif action_type == 2:  # Remove an existing ask
                self.asks[level].amount = 0

        self._sort_lob()

    def _simulate_market(self):
        num_market_orders = 1

        for _ in range(num_market_orders):
            if np.random.rand() < 0.5:
                if self.asks:
                    best_ask = next((order for order in self.asks if order.amount > 0), None)
                    if best_ask:
                        trade_amount = min(best_ask.amount, np.random.randint(1000, 5000))  # random or available amount
                        self.inventory -= trade_amount
                        self.capital += trade_amount * best_ask.price
                        best_ask.amount -= trade_amount
            else:
                if self.bids:
                    best_bid = next((order for order in self.bids if order.amount > 0), None)
                    if best_bid:
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
        inventory_penalty = 0.0001 * np.abs(self.inventory - self.initial_position)

        spread_penalty = 0
        for bid, ask in zip(self.bids, self.asks):
            spread = ask.price - bid.price
            if spread / bid.price > 0.02:
                spread_penalty += 0.01 * spread

        zero_amount_penalty = sum(5 for order in self.bids + self.asks if order.amount == 0)

        return 1 - inventory_penalty - spread_penalty - zero_amount_penalty

# Register the custom environment
gym.register(
    id='MarketMaking-v1',
    entry_point=MarketSimEnv,
)

if __name__ == "__main__":
    env = gym.make('MarketMaking-v1')
    check_env(env)

    model = DQN('MlpPolicy', env, verbose=1, exploration_fraction=0.2)
    model.learn(total_timesteps=100_000)

    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()