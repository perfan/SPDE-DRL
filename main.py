import matplotlib.pyplot as plt 
import numpy as np
from SpdeEnv import SpdeEnv
from SPDEs import Burgers
from math import pi as PI
from utils import *
from model import Agent
import os
import time
from datetime import datetime


config = Config()
config.n_t = 2
config.t_start = 0
config.t_end = 2
config.n_x = 151
config.x_max = 2.0*PI
config.nu = 0.01
config.eps = 0.01 #0.01
config.regularizer_weight = 0.2
config.func_type = "piece-wise-constant"
config.u_max = 8
config.f_max = 10

config.alpha = 0.000025
config.beta = 0.00025
config.batch_size=64
config.layer1_size=400
config.layer2_size=300
config.num_episodes = 100
config.episode_length = 1201

seed = 0
np.random.seed(seed)

burgers = Burgers(XMAX, NX, NT)


if func_type == "piece-wise-constant":
    config.theta_high = np.array([1, config.n_x, config.f_max, config.f_max], dtype = np.float32)
    sconfig.theta_scale = [1, 1, 1, 1]
    config.theta_size = 4
elif func_type == "linear-tanh":
    config.theta_high = np.array([config.f_max, config.f_max, config.f_max, config.f_max], dtype = np.float32)
    config.theta_scale = [1, 50, 1, 1]
    config.theta_size = 4
else:
    config.theta_high = np.full((config.n_x), config.f_max, dtype = np.float32)
    config.theta_scale = np.ful((config.n_x), 1,  dtype = np.float32)
    config.theta_size = config.n_x



env = SpdeEnv(burgers, config)
chkpt_dir = 'experiment_out'
make_dir(chkpt_dir)

agent = Agent(alpha= config.alpha, beta=config.beta, input_dims=[config.n_x], tau=config.tau, env=env,
              batch_size= config.batch_size,  layer1_size=config.layer1_size, layer2_size=config.layer2_size, n_actions= self.theta_size, chkpt_dir=chkpt_dir)


make_dir('logs')
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
log_dir_name = "./logs/{}".format(current_time)
os.mkdir(log_dir_name)

score_history = []

timeStep = (config.t_end - config.t_start) / (config.episode_length-1)
for j in range(config.num_episodes):
    obs = env.reset()
    done = False
    score = 0

    now = datetime.now()
    iter_log_dir_name = "{}/{}".format(log_dir_name, j)
    os.mkdir(iter_log_dir_name)
    for i in range(config.episode_length):
        t_init = i * timeStep
        t_final = (i + 1) * timeStep

        act = agent.choose_action(obs).astype('double')
        act = act * theta_scale
        
        idx = np.argmax(obs)
        
        if i%100 == 0 :
            if config.func_type == "piece-wise-constant":
                f = env.piece_wise_constant(act)
            elif config.func_type == "linear-tanh":
                f = env.tanh_linear_comp(act)
            else:
                f = act

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
    score = score / config.episode_length
    score_history.append(score)

    print('episode ', j, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = '{}-alpha{}-beta{}.png'.format(config.func_type, config.alpha, config.beta)
plotLearning(score_history, filename, window=100)
