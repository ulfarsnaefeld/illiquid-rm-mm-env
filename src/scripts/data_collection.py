import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulators import MarketSimEnv

params = {
    'sigma': [0.05, 0.1, 0.15],
    'drift': [0.001, 0.005, 0.01],
    'gamma': [0.99, 0.95, 0.9],
    'itp': [0.5, 0.1, 0.2],
    'ms': [0.01, 0.03, 0.05],
    'tf': [0.5, 1, 2],
    'inventory': [0, 100, 500]
}

def run_simulation_and_save(params, param_type, param_value, model_type='DQN'):
    model_path = f'src/models/trained_models/{model_type}/{param_type}/{param_value}/market_making_{model_type}'

    model = (
        A2C.load(model_path) if model_type == 'A2C' else
        PPO.load(model_path) if model_type == 'PPO' else
        DQN.load(model_path)
    )

    env = MarketSimEnv(**params)

    data_path = f'src/data/{model_type}/{param_type}/{param_value}'
    os.makedirs(data_path, exist_ok=True)

    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render(folder_path=f'src/data/{model_type}/{param_type}/{param_value}/')


def run_all_simulations():
    base_params = {'s0': 100, 'T': 100, 'v': 1, 'q0': 0, 'c0': 100_000, 'sigma': 0.1, 'drift': 0.005, 'gamma': 0.95, 'itp': 0.1, 'ms': 0.03, 'tf': 1}

    illiquid_params = {'s0': 100, 'T': 100, 'v': 1, 'q0': 0, 'c0': 100_000, 'sigma': 0.15, 'drift': 0.005, 'gamma': 0.95, 'itp': 0.2, 'ms': 0.03, 'tf': 0.5}

    # Different types of RL models provided by stable_baselines3
    model_types = ["DQN", "PPO", "A2C"]

    for sigma in params['sigma']:
        params_copy = base_params.copy()
        params_copy['sigma'] = sigma
        for model_type in model_types:
            run_simulation_and_save(params_copy, 'sigma', sigma, model_type)

    for drift in params['drift']:
        params_copy = base_params.copy()
        params_copy['drift'] = drift
        for model_type in model_types:
            run_simulation_and_save(params_copy, 'drift', drift, model_type)

    for gamma in params['gamma']:
        params_copy = base_params.copy()
        params_copy['gamma'] = gamma
        for model_type in model_types:
            run_simulation_and_save(params_copy, 'gamma', gamma, model_type)

    for itp in params['itp']:
        params_copy = base_params.copy()
        params_copy['itp'] = itp
        for model_type in model_types:
            run_simulation_and_save(params_copy, 'itp', itp, model_type)

    for ms in params['ms']:
        params_copy = base_params.copy()
        params_copy['ms'] = ms
        for model_type in model_types:
            run_simulation_and_save(params_copy, 'ms', ms, model_type)

    for tf in params['tf']:
        params_copy = base_params.copy()
        params_copy['tf'] = tf
        for model_type in model_types:
            run_simulation_and_save(params_copy, 'tf', tf, model_type)

    for inventory in params['inventory']:
        params_copy = base_params.copy()
        params_copy['q0'] = inventory
        for model_type in model_types:
            run_simulation_and_save(params_copy, 'inventory', inventory, model_type)

    for model_type in model_types:
        run_simulation_and_save(base_params, 'liquid', '', model_type)

    for model_type in model_types:
        run_simulation_and_save(illiquid_params, 'illiquid', '', model_type)


if __name__ == "__main__":
    run_all_simulations()