import matplotlib.pyplot as plt 
import numpy as np
from SpdeEnv import SpdeEnv
from SPDEs import Burgers
from math import pi as PI
from utils import *



config = Config()
config.n_t = 1201
config.t_start = 0
config.t_end = 2
config.n_x = 151
config.x_max = 2.0 * PI
config.nu = 0.01
config.eps = 0.01
config.regularizer_weight = 0
config.func_type = "linear"
config.cost_type = "variance"
config.u_max = 8
config.f_max = 10

np.random.seed(0)

burgers = Burgers(config.x_max, config.n_x, config.n_t)

config.theta_high = np.full((151), config.f_max, dtype = np.float32)
# f_control = np.concatenate([ np.full((75), -2),  np.full((76), 2)])
f_control = np.full((151), 0)
print(f_control)



env = SpdeEnv(burgers, config)
prev_condition = env.reset()

u, du, reward, done, _ = env.step(f_control, config.t_start, config.t_end, prev_condition)

print(reward)

plt.figure(figsize=(10,7))

plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rcParams['axes.linewidth'] = 2

plt.xlabel('x', fontsize = 10)
plt.ylabel(r'$u$', fontsize = 10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)  

plt.plot(burgers.x, env.state[:,0], label=r'$t=0$(s)', color="black", lw=2)
plt.plot(burgers.x, env.state[:,200], label=r'$t=16$(s)', color="red", lw=2)
plt.plot(burgers.x, env.state[:,400], label=r'$t=0.33$(s)', color="green", lw=2)
plt.plot(burgers.x, env.state[:,600], label=r'$t=0.50$(s)', color="blue", lw=2)

plt.xlim(0, config.x_max)
# plt.ylim(0, 7.5)
plt.legend(prop={'size': 10})
plt.minorticks_on()    
plt.savefig("test-some-control-2")
plt.show()
