import gymnasium as gym
import numpy as np
import math
# make the simulation into an RL environment:
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from runstats import *
import runstats

np.random.seed(1234)

actions_num = 21   #MS: So the range of possibilities goes from 0.3% to 3% from TOB
max_abs_dif = 4
max_abs_spread = 20


s0 = 100
T = 1. # Total time.
sigma = 2.  # Standard deviation.
dt = .005  # Time step.
beta = 0.5
kappa = beta * 2
k = 1.5
A = 137.45

def spread(beta, sigma, T_t, k):
    return beta*sigma**2*(T_t) + 2/beta*np.log(1+beta/k)

def r(beta, sigma, T_t, s, q):
    return s - q*beta*sigma**2*(T_t)

def l(A, k, d):
    '''
    Parameters
    ----------
      A : float
        in Avellaneda A = \lambda/\alpha, where alpha is as above,
        and lambda is the constant frequency of market buy and sell orders.
      k : float
        in Avellaneda k = alpha*K, where alpha ~ 1.5,
        and K is such that \delta p ~ Kln(Q) for a market order of size Q
      d : float
        in Avellaneda, d=distance to the mid price

    Return
    -------

      l : float:
        in Avellaneda, l = lambda = Poisson intensity at which our agent’s orders are
        executed.
    '''
    return A*np.exp(-k*d)
    #JK: eq. (12)


class AvellanedaEnv(gym.Env):
    def __init__(self, s0, T, dt, sigma, beta, k, A, kappa, seed=0, is_discrete=True):
        '''
        Parameters
        ----------
        s : float
            Initial value of future/stock price.
        b : float
            Initial value of 'brecha'.
        T : float
            Total time.
        dt : float
            Time subdivision.
        sigma : float
            price volatility.
        gamma : float
            discount factor.
        k : float
            in Avellaneda k = alpha*K, where alpha ~ 1.5,
            and K is such that \delta p ~ Kln(Q) for a market order of size Q
        A : float
            in Avellaneda A = \lambda/\alpha, where alpha is as above,
            and lambda is the constant frequency of market buy and sell orders.

        '''
        self.s0 = s0
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.beta = beta
        self.k = k
        self.A = A
        self.sqrtdt = np.sqrt(dt)
        self.kappa = kappa
        self.is_discrete = is_discrete
        self.stats = runstats.ExponentialStatistics(decay=0.999)
        np.random.seed(seed)

        # observation space: s (price), q, T-t (time remaining)
        self.observation_space = gym.spaces.Box(low=np.array([0.0, -math.inf, 0.0]),
            high=np.array([math.inf, math.inf,T]),
            dtype=np.float32)
        # action space: spread, ds
        self.action_space = gym.spaces.Discrete(actions_num)
        self.reward_range = (-math.inf,math.inf)

        self.metadata = None # useless field

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.s = self.s0
        self.q = 0.0
        self.t = 0.0
        self.w = 0.0
        self.n = int(T/dt)
        self.c_ = 0.0

        observation = np.concatenate([[self.s], [self.q], [self.T]])
        return observation.astype(np.float32), {'w':self.w}

    def _get_observation(self):
        observation = np.concatenate([[self.s], [self.q], [self.T]])
        return observation.astype(np.float32)

    def step(self, action):
        if self.is_discrete:
            despl = (action-(actions_num-1)/2)*max_abs_dif/(actions_num-1)
        else:
            despl = action
        ba_spread = spread(self.beta,self.sigma,self.T-self.t,self.k)

        bid = self.s - despl - ba_spread/2
        ask = self.s - despl + ba_spread/2

        self.bid = bid
        self.ask = ask

        db = self.s - bid
        da = ask - self.s

        lb = l(A, k, db)
        la = l(A, k, da)

        dnb = 1 if np.random.uniform() <= lb * self.dt else 0
        dna = 1 if np.random.uniform() <= la * self.dt else 0
        self.q += dnb - dna

        self.c_ += -dnb * bid + dna * ask # cash

        self.s += self.sigma * self.sqrtdt *(1 if np.random.uniform() < 0.5 else -1)

        previous_w = self.w
        self.w = self.c_ + self.q * self.s

        dw = (self.w - previous_w)
        self.stats.push(dw)
        #reward =  np.exp(-self.gamma*previous_w) - np.exp(-self.gamma*self.w) - 1/(self.n)

        #if self.t >= self.T:
        reward = dw - self.kappa/2 * (dw - self.stats.mean())**2

        #if self.t >= self.T - self.dt:
            #print("sum of dw: " + str(sum(self.ws)))
            #print("sum of kappa/2 * (dw - mu)**2: " + str(sum(self.rews)))

        self.t += self.dt
        return self._get_observation(), reward, self.t >= self.T, False, {'w':self.w}

    def render(self):
        print(f"Bid: {self.bid}, Mid: {self.s}, Ask: {self.ask}, Cash: {self.s}, Inventory: {self.q}")

if __name__ == "__main__":
    env = AvellanedaEnv(s0, T, dt, sigma, beta, k, A,kappa)
    check_env(env)

    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=50_000)

    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()



