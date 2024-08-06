import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from simulators import SeededMarketSimEnv

# Define the parameter sets
params = {
    'sigma': [0.05, 0.1, 0.15],
    'drift': [0.001, 0.005, 0.01],
    'gamma': [0.99, 0.95, 0.9],
    'itp': [0.5, 0.1, 0.2],
    'ms': [0.01, 0.03, 0.05],
    'tf': [0.5, 1, 2],
    'inventory': [0, 100, 500],
    'itfp': [1, 10, 50]
}

# Function to train the model and save it
def train_and_save_model(params, param_type, param_value, current_time, model_type='DQN'):
    log_dir = os.path.join('src/models', 'trained_models', current_time, model_type, param_type, str(param_value))
    os.makedirs(log_dir, exist_ok=True)

    env = SeededMarketSimEnv(**params)
    env = Monitor(env, log_dir)

    # Check if the environment is valid
    check_env(env)

    model = (
        A2C("MlpPolicy", env, verbose=1) if model_type == 'A2C' else
        PPO("MlpPolicy", env, verbose=1) if model_type == 'PPO' else
        DQN("MlpPolicy", env, verbose=1))

    # Train the model
    model.learn(total_timesteps=200_000)

    # Save the model
    model.save(os.path.join(log_dir, f'market_making_{model_type}'))

    print(f"Model trained and saved for {param_type}={param_value} on {model_type}")

# Train models for different parameters
def train_all_models():
    current_time = datetime.now().strftime('%d-%m-%Y_%H:%M')

    seeds_df = pd.read_csv('seeds.csv')
    seed_sequence = seeds_df['seed'].tolist()

    base_params = {'s0': 100, 'T': 400, 'v': 1, 'q0': 0, 'c0': 100_000, 'sigma': 0.1, 'drift': 0.01, 'gamma': 0.99, 'itp': 0.2, 'ms': 0.03, 'tf': 1, 'itfp': 10, 'seed_sequence': seed_sequence}
    illiquid_params = {'s0': 100, 'T': 400, 'v': 1, 'q0': 0, 'c0': 100_000, 'sigma': 0.2, 'drift': 0.01, 'gamma': 0.99, 'itp': 0.5, 'ms': 0.03, 'tf': 0.5, 'itfp': 10 ,'seed_sequence': seed_sequence}

    # Different types of RL models provided by stable_baselines3
    model_types = ["DQN", "PPO", "A2C"]

    # for sigma in params['sigma']:
    #     params_copy = base_params.copy()
    #     params_copy['sigma'] = sigma
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'sigma', sigma, current_time, model_type)

    # for drift in params['drift']:
    #     params_copy = base_params.copy()
    #     params_copy['drift'] = drift
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'drift', drift, current_time, model_type)

    # for gamma in params['gamma']:
    #     params_copy = base_params.copy()
    #     params_copy['gamma'] = gamma
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'gamma', gamma, current_time, model_type)

    # for itp in params['itp']:
    #     params_copy = base_params.copy()
    #     params_copy['itp'] = itp
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'itp', itp, current_time, model_type)

    # for ms in params['ms']:
    #     params_copy = base_params.copy()
    #     params_copy['ms'] = ms
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'ms', ms, current_time, model_type)

    # for tf in params['tf']:
    #     params_copy = base_params.copy()
    #     params_copy['tf'] = tf
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'tf', tf, current_time, model_type)

    # for inventory in params['inventory']:
    #     params_copy = base_params.copy()
    #     params_copy['q0'] = inventory
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'inventory', inventory, current_time, model_type)

    # for itfp in params['itfp']:
    #     params_copy = base_params.copy()
    #     params_copy['itfp'] = itfp
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'itfp', itfp, current_time, model_type)

    # for itfp in params['itfp']:
    #     params_copy = base_params.copy()
    #     params_copy['itfp'] = itfp
    #     for model_type in model_types:
    #         train_and_save_model(params_copy, 'itfp', itfp, current_time, model_type)

    for model_type in model_types:
        train_and_save_model(base_params, 'liquid', '', current_time, model_type)

    for model_type in model_types:
        train_and_save_model(illiquid_params, 'illiquid', '', current_time, model_type)

if __name__ == "__main__":
    for i in range (5):
        train_all_models()