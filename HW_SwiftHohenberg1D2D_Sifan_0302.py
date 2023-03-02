# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:30:42 2023

@author: Sifan Yin

Description: 
   spectral method for one variable Swift-Hohenberg equation in 1D and 2D
  
       dudt = mu*u-(1+Laplacian)^2*u+s*u^2-u^3
      

   space (1D, 2D): spectral interpolation (periodic BCs)
   time      : semi-implicit second-order Adams-Bashforth, Crank-Nicholson
    
"""
import os 
import numpy as np
import scipy as sp 
import random
import matplotlib.pyplot as plt
#import imageio
from numpy.fft import fft,ifft,fft2,ifft2
from matplotlib.pyplot import figure



#--------------------------------------------------------
# Functions    
def make_animation(filenames,output_path):
    with imageio.get_writer(output_path, mode = 'I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
def aap(uh,vh):
    N = len(uh)
    M = int(N*3/2)
    uh1 = uh[0:int(N/2)];uh2 = np.zeros(int(M-N));uh3 = uh[int(N/2):]
    uhp = np.concatenate((uh1,uh2,uh3))
    vh1 = vh[0:int(N/2)];vh2 = np.zeros(int(M-N));vh3 = vh[int(N/2):]
    vhp = np.concatenate((vh1,vh2,vh3))
    up = ifft(uhp)
    vp = ifft(vhp)
    w = up*vp
    wh = fft(w)
    ph1 = wh[0:int(N/2)];ph2 = wh[M-int(N/2):M+1]
    ph = np.concatenate((ph1,ph2))
    return ph
    



            
#------------------------------------------------
# Computing domain discretization
# Grid and required matrices:
L = 10*np.pi
N = 512
x = L/N*np.arange(0,N)
X,Y = np.meshgrid(x,x)
kx = 2*np.pi/L*np.concatenate((np.arange(0,N/2),np.arange(-N/2,0,1)))
[KX,KY] = np.meshgrid(kx,kx)
k2 = kx*kx
k4 = k2*k2

K2 = KX*KX+KY*KY
K4 = K2*K2
#%%
TotalTime = 100
dt = 0.02
time = np.arange(0,TotalTime+dt,dt)
tsteps = int(TotalTime/dt+1)  # number of timesteps
time_gap = 10

theta = 0.5 #weight to the current time-step value for the linear operator. theta = 0 => implicit, theta = 1 =>explicit, theta = 0.5 => Crank-Nicholson 

filenames = ['SH1D_Time_'+str(time_point).zfill(3)+'.png' for time_point in range(int(TotalTime/time_gap)+1)]


#----------------------
# Parameters
mu = 0.7
s = 1
# Differential operator matrix
Lop = (mu-1)+2.0*k2-k4
LOP = (mu-1)+2.0*K2-K4

Left_mat = np.ones(N)-(1.0-theta)*dt*Lop  #1D
Right_mat = np.ones(N) + theta*dt*Lop

LEFT_MAT = np.eye(N) -(1.0-theta)*dt*LOP  #2D
RIGHT_MAT = np.eye(N) +(theta*dt)*LOP




# Initial condition
u0 = 0.2*(np.random.random(N)-0.5)+0.5
usol = u0.copy() 

u = u0                    
uh = fft(u)
Nu = s*u*u-u*u*u
Nuh = fft(Nu)
j = 0
#%%
#time stepping
for i in np.arange(tsteps+1):
    if i%int(time_gap/dt) ==0:
        #plt.ylim((-1,1.1))
        plt.xlabel('x');plt.ylabel('u')
        plt.plot(x,np.real(ifft(uh)))
        plt.title('Time = '+str(time[i]).zfill(5))
        plt.show()
        plt.savefig(filenames[j],dpi = 300)
        j = j+1
        plt.clf()
    Nuh_old = Nuh
    Nuh = s*aap(uh,uh)-aap(uh,aap(uh,uh))
    #uh_new = Left_mat**(-1)*(Right_mat*uh + 1.5*dt*Nuh - 0.5*dt*Nuh_old)
    uh_new = 1/Left_mat*(Right_mat*uh+dt*Nuh)
    #uh_new = np.linalg.solve(Left_mat,Right_mat*uh+1.5*dt*Nuh-0.5*dt*Nuh_old)
    uh = uh_new


output_path = './SwiftHohenberg1D_simulation.gif'        
make_animation(filenames,output_path)

