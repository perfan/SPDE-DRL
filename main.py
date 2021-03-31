from math import pi as PI
import numpy as np
import Burgers
import matplotlib.pyplot as plt

seed  = 0

NT = 301
T_START = 0
T_END = 1
NX = 151
XMAX = 2.0*PI
NU = 0.01
EPS = 0.01

np.random.seed(seed)
u = np.zeros((NX,NT))
x = np.linspace(0, XMAX, NX)
t = np.linspace(T_START, T_END, NT)

prev_condition = Burgers.InitialCondition(u, x, NX, NU)
u = Burgers.convection_diffusion(u, x, t, NT, NX, T_START, T_END, XMAX, NU, EPS, prev_condition)

plt.figure(figsize=(10,7))

plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rcParams['axes.linewidth'] = 2

plt.xlabel('x', fontsize = 10)
plt.ylabel(r'$u$', fontsize = 10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)  

plt.plot(x, u[:,0], label="u_initial", color="black", lw=2)
plt.plot(x, u[:,100], label="u_t1", color="red", lw=2)
plt.plot(x, u[:,200], label="u_t2", color="green", lw=2)
plt.plot(x, u[:,-1], label="u_end", color="blue", lw=2)

plt.xlim(0, XMAX)
plt.ylim(0, 7.5)
plt.legend(prop={'size': 10})
plt.minorticks_on()    
plt.show()