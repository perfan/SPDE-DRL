import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import utils


class SpdeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def tanh_linear_comp(self, theta):
        config = self.config
        theta = np.clip(theta, -config.theta_high, config.theta_high)
        a = np.arange(config.n_x, dtype = np.float32)
        f = np.tanh(theta[0] * a + theta[1]) * theta[2] + theta[3]
        return f
    
    def piece_wise_constant(self, theta):
        config = self.config
        theta_size = len(theta)
        theta = np.clip(theta, -config.theta_high, config.theta_high)
        f = np.array([], dtype = np.float32)
        for i in range(config.theta_size):
          f = np.concatenate([f, np.full((config.n_x // theta_size),  theta[i], dtype = np.float32)])
        f = np.concatenate([f, np.full((config.n_x % theta_size),  theta[theta_size - 1], dtype = np.float32)])
        return f

    def regularize_f(self, f):
        f = np.clip(f, -self.config.f_max, self.config.f_max)
        return f - np.average(f)

    def __init__(self, burgers, config):
        self.burgers = burgers
        self.config = config
        
      
        u_high = np.full((config.n_x), config.u_max, dtype = np.float32)

        self.action_space = spaces.Box(
            low=-config.theta_high,
            high=config.theta_high,
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-u_high,
            high=u_high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, theta, t_start, t_end, prev_condition):
        config = self.config
        if config.func_type == "piece-wise-constant":
            f = self.piece_wise_constant(theta)
        elif config..func_type == "linear-tanh":
            f = self.tanh_linear_comp(theta)
        elif config.func_type == "linear":
            f = theta
        f = self.regularize_f(f)
        
        
        u = self.burgers.convection_diffusion(t_start, t_end, config.nu, config.eps, prev_condition, f)
        self.state = u
        # costs = np.sum((u[:, -1] - self.u_star)**2)
        # costs = np.sum((u[:, -1] - self.u_star)**2)
        du = utils.differentiate(u[:,-1], config.x_max, config.n_x)
        
        regularizer = config.regularizer_weight * np.average((f[1:] - f[:-1])**2)
        u_avg  = np.average(u[:,-1])
        costs = np.average((u[:, -1] -  u_avg)**2)
        # costs += regularizer
        # costs = np.max(u[:,-1]) - np.min(u[:,-1])

        return self.state[:, -1], du, -costs, False, {}

    def reset(self):
        self.state = self.burgers.u 
        self.state[:, 0] = self.burgers.InitialCondition(config.nu) 
        self.last_u = None
        return self.state[:, 0]
