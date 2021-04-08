import matplotlib.pyplot as plt 
import numpy as np
from SpdeEnv import SpdeEnv
from SPDEs import Burgers
from math import pi as PI
from utils import make_dir
from utils import plotLearning
from model import Agent
import os
import time
from datetime import datetime


NT = 2
T_START = 0
T_END = 2
NX = 151
XMAX = 2.0*PI
NU = 0.01
EPS = 0.01 #0.01
seed  = 0
lambda1 = 0.2


burgers = Burgers(XMAX, NX, NT)
UMAX = 8
F_MAX = 10
USTAR = np.full(NX, 4.0, dtype = np.float32)
THETAHIGH = np.array([1, NX, F_MAX, F_MAX], dtype = np.float32)
THETASIZE = 4
theta_scale = [1, 75, 1, 1]

env = SpdeEnv(burgers, UMAX, F_MAX, THETAHIGH, THETASIZE, NU, EPS, USTAR, lambda1)
chkpt_dir = 'experiment_out'
make_dir(chkpt_dir)

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[NX], tau=0.1, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions= THETASIZE, chkpt_dir=chkpt_dir)

np.random.seed(seed)

make_dir('logs')
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
log_dir_name = "./logs/{}".format(current_time)
os.mkdir(log_dir_name)

score_history = []
num_episodes = 100
episode_length = 1201
timeStep = (T_END-T_START)/(episode_length-1)
for j in range(num_episodes):
    obs = env.reset()
    done = False
    score = 0

    now = datetime.now()
    iter_log_dir_name = "{}/{}".format(log_dir_name, j)
    os.mkdir(iter_log_dir_name)
    for i in range(episode_length):
        t_init = i * timeStep
        t_final = (i + 1) * timeStep

        act = agent.choose_action(obs).astype('double')
        act = act * theta_scale
        
        idx = np.argmax(obs)
        
        if i%100 == 0 :
          f = env.function_theta(act)
          plt.plot(f)
          plt.savefig("{}/{}-act.png".format(iter_log_dir_name, i))
          plt.close()

          plt.plot(obs)
          plt.savefig("{}/{}-obs.png".format(iter_log_dir_name, i))
          plt.close()

        new_state, derivaties, reward, done, info = env.step(act, t_init, t_final, obs)
        
        if i%100 == 0 :
          plt.plot(new_state)
          plt.savefig("{}/{}-newState.png".format(iter_log_dir_name, i))
          plt.close()

        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

        #env.render()

    score_history.append(score / episode_length)

    print('episode ', j, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)
