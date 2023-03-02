# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:09:56 2023

@author: Sifan Yin

Description: finite difference method for FKPP equation
   space: second order centeral difference method
   time: implicit Crank-Nicolson method
    
    
"""
import os 
import numpy as np 
from scipy import sparse
from math import e
import matplotlib.pyplot as plt
#import imageio

np.set_printoptions(precision=3)


def make_animation(filenames,output_path):
    with imageio.get_writer(output_path, mode = 'I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
  
#-----------------------------
#   Parameters
#-----------------------------
L = 400.   #computing domain 
N = 201    #number of grids
dx = L/(N-1)
X = np.arange(-L/2,L/2+dx,dx)
width_ratio = 0.1

TotalTime = 10
dt = 0.01
tsteps = int(TotalTime/dt)+1
Time = np.arange(0,TotalTime+dt,dt)
time_gap = 1


D = 0.1
r = 1


output_path = './FKPP1D_simulation.gif'
filenames = ['FKPP1D_Time_'+str(time_point).zfill(3)+'.png' for time_point in range(int(TotalTime/time_gap)+1)]




# Initial conditions
IC = 1 # Gaussian 
#IC = 2 # square wave
#BC = 1 # Periodic
BC = 2 # Neumann
#BC = 3 # Dirichlet


x0 = 0
u_max = 0.3
u_min = 0
u_ampt = u_max-u_min
width = width_ratio*L
width_point_side = int((L/2-width)/dx)
width_point_center = int(width/dx)
if IC==1:
   #Case 1:Gussian distribution for u
    u0 = u_ampt*pow(e,-(X-x0)**2/(width**2))+u_min
else:
    #Case 2: square wave for u
    u0 = np.array([u_min for i in range(0,width_point_side)]+[u_max for i in range(0,2*width_point_center+1)]+[u_min for i in range(0,width_point_side)])
    


#------------------------------
# Differential matrix
#------------------------------
sigma_u = D*dt/(2.*dx**2)

if BC==1: 
    #Periodic BC
    DiffMatrix_left_u  = sparse.diags([-sigma_u,-sigma_u,1+2*sigma_u,-sigma_u,-sigma_u],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_right_u = sparse.diags([ sigma_u, sigma_u,1-2*sigma_u, sigma_u, sigma_u],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
  


elif BC==2:
    #Neumann BC (no flux)
    DiffMatrix_left_u = sparse.diags([-sigma_u,-sigma_u,1+2*sigma_u,-sigma_u,-sigma_u],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_left_u[0][0]= 1 + sigma_u;DiffMatrix_left_u[0][-1]=0.;DiffMatrix_left_u[-1][0] = 0.;DiffMatrix_left_u[-1][-1] = 1+sigma_u
   
    DiffMatrix_right_u = sparse.diags([ sigma_u, sigma_u,1-2*sigma_u, sigma_u, sigma_u],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_right_u[0][0]= 1 - sigma_u;DiffMatrix_right_u[0][-1]=0.;DiffMatrix_right_u[-1][0] = 0.;DiffMatrix_right_u[-1][-1] = 1-sigma_u
  
elif BC==3:
    #Dirichlet BC
    print(1)

#-------------------------
# Time stepping
#-------------------------
U = u0
Usol = [];#Usol.append(U)

j = 0
for i in range(tsteps):
    if i%int(time_gap/dt) ==0:
        plt.ylim((-1,1.1));plt.xlim((-L/2,L/2))
        plt.xlabel('X');plt.ylabel('Concentration')
        plt.plot(X,U)
        plt.title('Time = '+str(Time[i]).zfill(5))
        plt.show()
        plt.savefig(filenames[j],dpi = 300)
        j = j+1
        plt.clf()
    Nu = dt*r*U*(1.-U)   
    Unew = np.linalg.solve(DiffMatrix_left_u,DiffMatrix_right_u.dot(U) + Nu) 
    U = Unew  
    Usol.append(U)
 
    
#make_animation(filenames,output_path)        

Usol = np.array(Usol)
fig,ax = plt.subplots()
plt.xlabel('x');plt.ylabel('Time')
heatmap = ax.pcolor(X,Time,Usol,vmin = -1,vmax = 1.1)
plt.title('FKPP 1D kymograph')
plt.savefig('FKPP1D_kymograph_Sifan_0302.png',dpi = 300)
