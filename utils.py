import os
import matplotlib.pyplot as plt 
import numpy as np


def differentiate(u, XMAX, NX):
    du = np.zeros(np.size(u))
    DX = XMAX/(NX-1) 

    du[0] = (u[1]-u[0])/DX
    for i in range(1, np.size(u)-1):
        du[i] = (u[i-1]-2*u[i]+u[i+1])/(2*DX)
    du[np.size(u)-1] = (u[np.size(u)-1]-u[np.size(u)-2])/DX

    return du

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

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