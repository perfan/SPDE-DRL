from math import pi as PI
from math import exp as exp
import numpy as np
import matplotlib.pyplot as plt

def Noise(Dx):
    return np.random.normal(loc=0.0, scale=np.sqrt(Dx))

def InitialCondition(u, x, NU):
    for i in range(0,NX):
        phi = exp( -(x[i]**2)/(4*NU) ) + exp( -(x[i]-2*PI)**2 / (4*NU) )
        dphi = -(0.5*x[i]/NU)*exp( -(x[i]**2) / (4*NU) ) - (0.5*(x[i]-2*PI) / NU )*exp(-(x[i]-2*PI)**2 / (4*NU) )
        u[i,0] = -2*NU*(dphi/phi) + 4
        
    #   u[i,0] = np.sin(PI * x[i]/x[NX-1])


def convection_diffusion(u, x, t, NT, NX, T_START, T_END, XMAX, NU, EPS, prev_condition):
   """
   Returns the velocity field and distance for 1D non-linear convection-diffusion
   """

   # Increments
   DT = (T_END-T_START)/(NT-1)
   DX = XMAX/(NX-1)

   # Initialise data structures
   ipos = np.zeros(NX)
   ineg = np.zeros(NX)

   # Periodic boundary conditions
   for i in range(0,NX):
       ipos[i] = i+1
       ineg[i] = i-1

   ipos[NX-1] = 0
   ineg[0] = NX-1

   # Numerical solution
   for n in range(0,NT-1):
       for i in range(0,NX):
           dw = Noise(DX)
           u[i,n+1] = (u[i,n]-u[i,n]*(DT/DX)*(u[i,n]-u[int(ineg[i]),n])+ NU*(DT/DX**2)*(u[int(ipos[i]),n]-2*u[i,n]+u[int(ineg[i]),n]) + (EPS/DX) * dw)

   return u


if __name__=='__main__':

    
    seed  = 0

    NT = 301
    NX = 151
    T_START = 0
    T_END = 1
    XMAX = 2.0*PI
    NU = 0.01
    EPS = 0.01

    np.random.seed(seed)
    u = np.zeros((NX,NT))
    x = np.linspace(0, XMAX, NX)
    t = np.linspace(T_START, T_END, NT)

    prev_condition = InitialCondition(u, x, NU)
    u = convection_diffusion(u, x, t, NT, NX, T_START, T_END, XMAX, NU, EPS, prev_condition)

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