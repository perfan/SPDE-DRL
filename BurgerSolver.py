import matplotlib.pyplot as plt 
import numpy as np
from SpdeEnv import SpdeEnv
from SPDEs import Burgers
from math import pi as PI


NT = 601
T_START = 0
T_END = 0.5
NX = 151
XMAX = 2.0*PI
NU = 0.01
EPS = 0.00 # 0.01
seed  = 0

np.random.seed(seed)

burgers = Burgers(XMAX, NX, NT)
UMAX = 8
USTAR = np.full(NX, 0, dtype = np.float32)
FMAX = 20
f_control = np.full(NX, 0, dtype = np.float32) 


env = SpdeEnv(burgers, UMAX, FMAX, NU, EPS, USTAR)
prev_condition = env.reset()

u, du, reward, done, _ = env.step(f_control, T_START, T_END, prev_condition)

print(reward)

plt.figure(figsize=(10,7))

plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rcParams['axes.linewidth'] = 2

plt.xlabel('x', fontsize = 10)
plt.ylabel(r'$u$', fontsize = 10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)  

plt.plot(burgers.x, env.state[:,0], label="u_initial", color="black", lw=2)
plt.plot(burgers.x, env.state[:,200], label="u_t1", color="red", lw=2)
plt.plot(burgers.x, env.state[:,400], label="u_t2", color="green", lw=2)
plt.plot(burgers.x, env.state[:,-1], label="u_end", color="blue", lw=2)

plt.xlim(0, XMAX)
# plt.ylim(0, 7.5)
plt.legend(prop={'size': 10})
plt.minorticks_on()    
plt.show()
