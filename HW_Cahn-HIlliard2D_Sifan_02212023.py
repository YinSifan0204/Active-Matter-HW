# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:54:40 2023

@author: Sifan Yin

Description: finite difference method for Cahn-Hilliard equation 2D 
   space: second order centeral difference method
   time: first order forward Euler
"""
import os 
import numpy as np 
import matplotlib.pyplot as plt

from scipy import sparse
from scipy import linalg

import imageio

#----------------------------
#  Function definitions
#----------------------------
def Euler_forward(phi,dt,lap):
    '''
    Args:
        phi (np.array): phase concentration
        dt  (float): timestep
        lap (sparse.dia_matrix): laplacian operator with periodic BCs
    Returns:
        phi_n: phase concentration in the next timestep
    '''
    rhs = lap.dot(np.power(phi,3) - phi - (lap.dot(phi)))
    phi_n = phi + dt*rhs
    return phi_n

def make_animation(filenames,output_path):
    with imageio.get_writer(output_path, mode = 'I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
                      
#-------------------------------------------
# Input parameters
#-------------------------------------------
N = 200  #number of grids in each axis
dx = 1    #space discretization
dt = 0.01 #timestep
TotalTime = 500
tsteps = int(TotalTime/dt+1)  # number of timesteps

time_gap = 50
phi0 = 0  #initial average value of phi
noise = 0.1 # initial fluctuations of phi
seed = 0
output_path = './CH2D_simulation.gif'
filenames = ['CH2D_Time_'+str(time_point).zfill(3)+'.png' for time_point in range(int(TotalTime/time_gap)+1)]

#%%
#---------------------------
#  Initial conditions
#---------------------------
np.random.seed(seed = seed)
phi = phi0*np.ones((N,N)) + noise*(np.random.rand(N,N)-0.5)
t = np.arange(0,TotalTime+dt,dt)


#Conver the matrix aaray to col form
phi = np.ravel(phi, order = 'F')

# Laplacian operator
lap_1D = sparse.diags([1,1,-2,1,1],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray() 
#lap = lap_1D/(dx*dx)   
Iden = sparse.eye(N)
lap_2D = sparse.kron(Iden,lap_1D) + sparse.kron(lap_1D,Iden)
lap = sparse.dia_matrix(lap_2D)/(dx*dx)
# plt.matshow(lap.toarray())
# plt.show()

#---------------------------------------
# time stepping
#---------------------------------------
j = 0
for i in range(tsteps):
    # print(t[i])
    if i%int(time_gap/dt) == 0:
        print(t[i])
        phi_plot = phi.reshape((N,N),order = 'F')
        plt.imshow(phi_plot)
        plt.colorbar()
        plt.title('Time = '+str(t[i]).zfill(5))
        plt.savefig(filenames[j],dpi = 300)
        j = j+1
        plt.clf()
    phi = Euler_forward(phi,dt,lap)
 
make_animation(filenames,output_path)







