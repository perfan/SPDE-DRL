from math import pi as PI
from math import exp as exp
import numpy as np

def Noise(Dx):
    return np.random.normal(loc=0.0, scale=np.sqrt(Dx))

def InitialCondition(u, x, NX, NU):
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