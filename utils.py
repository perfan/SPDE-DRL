import os
import matplotlib.pyplot as plt 
import numpy as np
from math import pi as PI


def differentiate(u, XMAX, NX):
    du = np.zeros(np.size(u))
    DX = XMAX/(NX-1) 

    du[0] = (u[1]-u[0])/DX
    for i in range(1, np.size(u)-1):
        du[i] = (u[i-1]-2*u[i]+u[i+1])/(2*DX)
    du[np.size(u)-1] = (u[np.size(u)-1]-u[np.size(u)-2])/DX

    return du

def plotLearning(scores, filename, x=None, window=5, col='blue'):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg, color=col)
    plt.savefig(filename)
    plt.close()

def make_dir(path):
  try: os.mkdir(path)
  except: pass

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)


import argparse
import torch
import gym

# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/config.py
class Config:
    # DEVICE = torch.device('cpu')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.n_t = 2
        self.t_start = 0
        self.t_end = 2
        self.n_x = 151
        self.x_max = 2.0*PI
        self.nu = 0.01
        self.eps = 0.01 #0.01
        self.regularizer_weight = 0.2
        self.func_type = "linear"
        self.cost_type = "variance"
        self.u_max = 8
        self.f_max = 10
        self.theta_size = 4
        self.theta_high = None
        self.alpha = 0.000025
        self.beta = 0.00025
        self.batch_size=64
        self.layer1_size=400
        self.layer2_size=300
        self.num_episodes = 100
        self.episode_length = 1201  
        self.tau = 0.1

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
