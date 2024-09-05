# Market Making Simulator

This repository contains a market making simulator designed for evaluating reinforcement learning (RL) strategies in both liquid and illiquid market conditions. The simulator is built using the Gymnasium framework and allows for the testing of various RL algorithms such as Deep Q-Network (DQN), Advantage Actor-Critic (A2C), and Proximal Policy Optimization (PPO).

This simulator is part of a Master's thesis project at Reykjavik University

## Features

- **Simulation Environment:** Implements a stochastic price model with geometric Brownian motion (GBM).
- **Market Conditions:** Supports both liquid and illiquid market scenarios with adjustable parameters.
- **RL Algorithms:** Integrated support for most algorithms.
- **Market Maker Behavior:** Market makers (_The RL algorithm_) adjust bid-ask spread based on inventory and price.
- **Trader Models:** Simulates informed and uninformed traders to introduce market noise and adverse selection.

## Simulator Parameters

| Parameter                    | Liquid Value | Illiquid Value | Description                                            |
| ---------------------------- | ------------ | -------------- | ------------------------------------------------------ |
| Initial Stock Price          | 100          | 100            | Initial price of the stock                             |
| Price Volatility             | 0.050        | 0.025          | Volatility in price movements (GBM model)              |
| Trading Frequency            | 2.0          | 1.0            | Number of trades per timestep (Poisson process)        |
| Informed Trading Percentage  | 10%          | 20%            | Percentage of traders with insider information         |
| Maximum Spread               | 2%           | 2%             | The maximum spread allowed for the market maker quotes |
| Initial Inventory            | 0            | 0              | Starting position of the market maker                  |
| Initial Cash                 | 100,000      | 100,000        | Initial capital held by the market maker               |
| Price Drift                  | 0.005        | 0.005          | Drift in the stock price trend                         |
| Episode Length               | 200          | 200            | Total number of steps in each simulation episode       |
| Informed Trading Future Pred | 10           | 10             | Time steps ahead informed traders can see              |

## Reinforcement Learning Algorithms

This simulator currently supports three reinforcement learning algorithms:

1. **Deep Q-Network** (DQN)
2. **Advantage Actor-Critic** (A2C)
3. **Proximal Policy Optimization** (PPO)

Each algorithm is implemented using the Gymnasium interface and tested under different market conditions. The performance metrics used include Profit and Loss (PnL), Mean Absolute Position (MAP), and Skewness/Inventory correlation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
