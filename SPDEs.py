from math import pi as PI
from math import exp as exp
import numpy as np
import matplotlib.pyplot as plt 

class Burgers:

  def __init__(self, XMAX, NX, NT):
    self.XMAX = XMAX
    self.NX = NX
    self.NT = NT
    self.u = np.zeros((self.NX,self.NT))
    self.x = np.linspace(0, self.XMAX, self.NX)

  @staticmethod  
  def Noise(Dx):
      return np.random.normal(loc=0.0, scale=np.sqrt(Dx))

  def InitialCondition(self, NU):
      u_init = np.zeros((self.NX))
      for i in range(0,self.NX):
          phi = exp( -(self.x[i]**2)/(4*NU) ) + exp( -(self.x[i]-2*PI)**2 / (4*NU) )
          dphi = -(0.5*self.x[i]/NU)*exp( -(self.x[i]**2) / (4*NU) ) - (0.5*(self.x[i]-2*PI) / NU )*exp(-(self.x[i]-2*PI)**2 / (4*NU) )
          u_init[i] = -2*NU*(dphi/phi) + 4
      
    #   plt.plot(u_init)
    #   plt.show()
      return u_init
          

  def convection_diffusion(self, T_START, T_END, NU, EPS, prev_condition, f_control):
    """
    Returns the velocity field and distance for 1D non-linear convection-diffusion
    """
    t = np.linspace(T_START, T_END, self.NT)

    # Increments
    DT = (T_END-T_START)/(self.NT-1)
    DX = self.XMAX/(self.NX-1)

    # Initialise data structures
    ipos = np.zeros(self.NX)
    ineg = np.zeros(self.NX)

    # Periodic boundary conditions
    for i in range(0,self.NX):
        ipos[i] = i+1
        ineg[i] = i-1

    ipos[self.NX-1] = 0
    ineg[0] = self.NX-1

    self.u[:,0] = prev_condition
    # Numerical solution
    for n in range(0,self.NT-1):
        for i in range(0,self.NX):
            dw = self.Noise(DX)
            self.u[i,n+1] = (self.u[i,n]-self.u[i,n]*(DT/DX)*(self.u[i,n]-self.u[int(ineg[i]),n])+ NU*(DT/DX**2)*(self.u[int(ipos[i]),n]-2*self.u[i,n]+self.u[int(ineg[i]),n]) + (EPS/DX) * dw) + f_control[i] * DT
    
    return self.u
