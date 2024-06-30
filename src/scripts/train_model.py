import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from simulators import MarketSimEnv

# Define the parameter sets
params = {
    'sigma': [0.05, 0.1, 0.15],
    'drift': [0.001, 0.005, 0.01],
    'gamma': [0.99, 0.95, 0.9],
    'itp': [0.5, 0.1, 0.2],
    'ms': [0.01, 0.03, 0.05],
    'tf': [0.5, 1, 2],
    'inventory': [0, 100, 500]
}

# Function to train the model and save it
def train_and_save_model(params, param_type, param_value):
    log_dir = os.path.join('src/models', 'trained_models', 'DQN', param_type, str(param_value))
    os.makedirs(log_dir, exist_ok=True)

    env = MarketSimEnv(**params)
    env = Monitor(env, log_dir)

    # Check if the environment is valid
    check_env(env)

    # Train the model
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    # Save the model
    model.save(os.path.join(log_dir, f'market_making_DQN'))

    print(f"Model trained and saved for {param_type}={param_value}")

# Train models for different parameters
def train_all_models():
    base_params = {'s0': 100, 'T': 100, 'v': 1, 'q0': 0, 'c0': 100_000, 'sigma': 0.1, 'drift': 0.005, 'gamma': 0.95, 'itp': 0.1, 'ms': 0.03, 'tf': 1}

    illiquid_params = {'s0': 100, 'T': 100, 'v': 1, 'q0': 0, 'c0': 100_000, 'sigma': 0.15, 'drift': 0.005, 'gamma': 0.95, 'itp': 0.2, 'ms': 0.03, 'tf': 0.5}

    for sigma in params['sigma']:
        params_copy = base_params.copy()
        params_copy['sigma'] = sigma
        train_and_save_model(params_copy, 'sigma', sigma)

    for drift in params['drift']:
        params_copy = base_params.copy()
        params_copy['drift'] = drift
        train_and_save_model(params_copy, 'drift', drift)

    for gamma in params['gamma']:
        params_copy = base_params.copy()
        params_copy['gamma'] = gamma
        train_and_save_model(params_copy, 'gamma', gamma)

    for itp in params['itp']:
        params_copy = base_params.copy()
        params_copy['itp'] = itp
        train_and_save_model(params_copy, 'itp', itp)

    for ms in params['ms']:
        params_copy = base_params.copy()
        params_copy['ms'] = ms
        train_and_save_model(params_copy, 'ms', ms)

    for tf in params['tf']:
        params_copy = base_params.copy()
        params_copy['tf'] = tf
        train_and_save_model(params_copy, 'tf', tf)

    for inventory in params['inventory']:
        params_copy = base_params.copy()
        params_copy['q0'] = inventory
        train_and_save_model(params_copy, 'inventory', inventory)

    train_and_save_model(base_params, 'liquid', '')

    train_and_save_model(illiquid_params, 'illiquid', '')

if __name__ == "__main__":
    train_all_models()