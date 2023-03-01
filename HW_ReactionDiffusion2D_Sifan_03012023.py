# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:30:42 2023

@author: Sifan Yin

Description: 
   finite difference method for two variables Reaction-diffusion equation in 2D
  
       dudt = Du*Lap(u)-u*v^2+F*(1-u)
       dvdt = Dv*Lap(v)+u*v^2-(F+k)*v

   space (2D): second order centeral difference method
   time      : implicit Crank-Nicolson method
    
"""
import os 
import numpy as np
import scipy as sp 
import random
import matplotlib.pyplot as plt
import imageio




#--------------------------------------------------------
# Functions    
def make_animation(filenames,output_path):
    with imageio.get_writer(output_path, mode = 'I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
            
#five-point stencil method of finite differences 
def laplacian_operator(U,V,dx):
    Lu = (U[0:-2,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] + U[2:,1:-1] - 4*U[1:-1,1:-1])/dx**2
    Lv = (V[0:-2,1:-1] + V[1:-1,0:-2] + V[1:-1,2:] + V[2:,1:-1] - 4*V[1:-1,1:-1])/dx**2
    return Lu,Lv

# Define parameters of Gray-Scott equation
pnames = ["solitons","coral","maze","waves","flicker","worms"]
pvals = [{"Du":0.14, "Dv":0.06, "F":0.035, "k":0.065,"Cmap":plt.cm.copper,"edgeMax":False}, \
         {"Du":0.16, "Dv":0.08, "F":0.06, "k":0.062,"Cmap":plt.cm.cubehelix,"edgeMax":False}, \
         {"Du":0.19, "Dv":0.05, "F":0.06, "k":0.062,"Cmap":plt.cm.cubehelix,"edgeMax":False}, \
         {"Du":0.12, "Dv":0.08, "F":0.02, "k":0.05,"Cmap":plt.cm.cubehelix,"edgeMax":True},\
         {"Du":0.16, "Dv":0.08, "F":0.02, "k":0.055,"Cmap":plt.cm.cubehelix,"edgeMax":True},\
         {"Du":0.16, "Dv":0.08, "F":0.054, "k":0.064,"Cmap":plt.cm.cubehelix,"edgeMax":False}]
pchoices = dict(zip(pnames,pvals))




#--------------------------------------------------------------
#Numerical simulation function for Gray-Scott equations
#--------------------------------------------------------------

# Computing domain 
L = 200
dx = 1
TotalTime = 10000
time_gap = 400
dt = 1
time = np.arange(0,TotalTime+dt,dt)
size = int(L/dx)

# Parameters
pattern = 'solitons'
pattern = 'coral'
pattern = 'waves'
pattern = 'maze'
pattern = 'worms'
pattern = 'flicker'

params = pchoices[pattern]
Du,Dv,k,F,Cmap,edgeMax = params['Du'],params['Dv'],params['k'],params['F'],params['Cmap'],params['edgeMax']


output_path = './GrayScott2D_'+pattern+'_Sifan_0301.gif'
filenames = ['GrayScott2D_'+pattern+'_Time_'+str(time_point).zfill(3)+'.png' for time_point in range(int(TotalTime/time_gap)+1)]

#%%

# Initial conditions
IC = 1 #single square
#IC = 2 #double square
#IC = 3 #random

U = np.zeros((size, size))
V = np.zeros((size, size))
u,v = U[1:-1,1:-1], V[1:-1,1:-1]
u+=1.0
#sets initialization of single or double squares, or completely random
if IC == 1:
    r = 20
    U[int(size/2-r):int(size/2+r),int(size/2-r):int(size/2+r)] = 0.5
    V[int(size/2-r):int(size/2+r),int(size/2-r):int(size/2+r)] = 0.25    
   
elif IC == 2:
    r = 15
    U[size/4-r:size/4+r,size/4-r:size/4+r] = 0.50
    V[size/4-r:size/4+r,size/4-r:size/4+r] = 0.25
    U[3*size/4-r:3*size/4+r,3*size/4-r:3*size/4+r] = 0.50
    V[3*size/4-r:3*size/4+r,3*size/4-r:3*size/4+r] = 0.25  
   
else: 
    u-=1
    u+=np.random.rand(len(u),len(u))
    v+=np.random.rand(len(u),len(u))

#%%
j = 0
for i in range(TotalTime+1):
    if i%time_gap==0:
        plt.imshow(v,cmap=Cmap,extent=[-1,1,-1,1])
        plt.axis('off')
        plt.colorbar()
        plt.title('Time = '+str(time[i]).zfill(5))
        plt.savefig(filenames[j],dpi = 300)
        j = j+1
        plt.clf()  
    Lu,Lv = laplacian_operator(U,V,dx)
    uvv = u*v*v
    su = Du*Lu - uvv + F *(1-u)
    sv = Dv*Lv + uvv - (F+k)*v
    u += dt*su
    v += dt*sv
    
            
make_animation(filenames,output_path)







