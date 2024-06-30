import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
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

def run_simulation_and_save(params, param_type, param_value):
    model_path = f'src/models/trained_models/DQN/{param_type}/{param_value}/market_making_DQN'
    model = DQN.load(model_path)

    env = MarketSimEnv(**params)

    data_path = f'src/data/DQN/{param_type}/{param_value}'
    os.makedirs(data_path, exist_ok=True)


    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render(folder_path=f'src/data/DQN/{param_type}/{param_value}/')


def run_all_simulations():
    base_params = {'s0': 100, 'T': 100, 'v': 1, 'q0': 0, 'c0': 100_000, 'sigma': 0.1, 'drift': 0.005, 'gamma': 0.95, 'itp': 0.1, 'ms': 0.03, 'tf': 1}

    illiquid_params = {'s0': 100, 'T': 100, 'v': 1, 'q0': 0, 'c0': 100_000, 'sigma': 0.15, 'drift': 0.005, 'gamma': 0.95, 'itp': 0.2, 'ms': 0.03, 'tf': 0.5}

    for sigma in params['sigma']:
        params_copy = base_params.copy()
        params_copy['sigma'] = sigma
        run_simulation_and_save(params_copy, 'sigma', sigma)

    for drift in params['drift']:
        params_copy = base_params.copy()
        params_copy['drift'] = drift
        run_simulation_and_save(params_copy, 'drift', drift)

    for gamma in params['gamma']:
        params_copy = base_params.copy()
        params_copy['gamma'] = gamma
        run_simulation_and_save(params_copy, 'gamma', gamma)

    for itp in params['itp']:
        params_copy = base_params.copy()
        params_copy['itp'] = itp
        run_simulation_and_save(params_copy, 'itp', itp)

    for ms in params['ms']:
        params_copy = base_params.copy()
        params_copy['ms'] = ms
        run_simulation_and_save(params_copy, 'ms', ms)

    for tf in params['tf']:
        params_copy = base_params.copy()
        params_copy['tf'] = tf
        run_simulation_and_save(params_copy, 'tf', tf)

    for inventory in params['inventory']:
        params_copy = base_params.copy()
        params_copy['q0'] = inventory
        run_simulation_and_save(params_copy, 'inventory', inventory)

    run_simulation_and_save(base_params, 'liquid', '')

    run_simulation_and_save(illiquid_params, 'illiquid', '')


if __name__ == "__main__":
    run_all_simulations()