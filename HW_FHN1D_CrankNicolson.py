# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:09:56 2023

@author: Sifan Yin

Description: finite difference method for FitzHugh-Nagumo (Reaction-diffusion) equation
   space: second order centeral difference method
   time: implicit Crank-Nicolson method
    
    
"""
import os 
import numpy as np 
from scipy import sparse
from math import e
import matplotlib.pyplot as plt
import imageio

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

TotalTime = 400
dt = 0.1
tsteps = int(TotalTime/dt)+1
Time = np.arange(0,TotalTime+dt,dt)
time_gap = 20


D_ratio = 1
Du = 1
Dv = D_ratio*Du

epsilon = 0.01
a = 0.1
b = 1; 
output_path = './FHN1D_simulation_b_'+str(b)+'.gif'
filenames = ['FHN1D_b'+str(b)+'_Time_'+str(time_point).zfill(3)+'.png' for time_point in range(int(TotalTime/time_gap)+1)]


# f = lambda u,v: u-u**3-v
# g = lambda u,v:epsilon*(u-b*v+a)
# f_col = lambda U,V: np.multiply(dt,np.subtract)

t1 = np.array([1,2,3],dtype=float)
t2 = np.array([4,5,6],dtype = float)
M = np.array([[1,0,-1],[0,2,0],[3,2,1]])
f = M.dot(t1)
tnew = np.linalg.solve(M,f)



# Initial conditions
IC = 1 # Gaussian 
#IC = 2 # square wave
#BC = 1 # Periodic
BC = 2 # Neumann
#BC = 3 # Dirichlet


x0 = 0
u_max = 1
u_min = -0.6
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
    
#tt = np.array([u_min for i in range(0,width_point)])

v_ampt = -0.3
v0 = v_ampt*np.ones(N)

# plt.ylim((-1,1.1))
# plt.xlabel('X')
# plt.ylabel('Concentration')
# plt.plot(X,u0)
# plt.plot(X,v0)
# plt.show()




#%%
#------------------------------
# Differential matrix
#------------------------------
sigma_u = Du*dt/(2.*dx**2)
sigma_v = Dv*dt/(2.*dx**2)

if BC==1: 
    #Periodic BC
    DiffMatrix_left_u  = sparse.diags([-sigma_u,-sigma_u,1+2*sigma_u,-sigma_u,-sigma_u],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_left_v  = sparse.diags([-sigma_v,-sigma_v,1+2*sigma_v,-sigma_v,-sigma_v],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_right_u = sparse.diags([ sigma_u, sigma_u,1-2*sigma_u, sigma_u, sigma_u],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_right_v = sparse.diags([ sigma_v, sigma_v,1-2*sigma_v, sigma_v, sigma_v],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()



elif BC==2:
    #Neumann BC (no flux)
    DiffMatrix_left_u = sparse.diags([-sigma_u,-sigma_u,1+2*sigma_u,-sigma_u,-sigma_u],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_left_u[0][0]= 1 + sigma_u;DiffMatrix_left_u[0][-1]=0.;DiffMatrix_left_u[-1][0] = 0.;DiffMatrix_left_u[-1][-1] = 1+sigma_u
    DiffMatrix_left_v = sparse.diags([-sigma_v,-sigma_v,1+2*sigma_v,-sigma_v,-sigma_v],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_left_v[0][0]= 1 + sigma_v;DiffMatrix_left_v[0][-1]=0.;DiffMatrix_left_v[-1][0] = 0.;DiffMatrix_left_v[-1][-1] = 1+sigma_v
    
    DiffMatrix_right_u = sparse.diags([ sigma_u, sigma_u,1-2*sigma_u, sigma_u, sigma_u],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_right_u[0][0]= 1 - sigma_u;DiffMatrix_right_u[0][-1]=0.;DiffMatrix_right_u[-1][0] = 0.;DiffMatrix_right_u[-1][-1] = 1-sigma_u
    DiffMatrix_right_v = sparse.diags([ sigma_v, sigma_v,1-2*sigma_v, sigma_v, sigma_v],[-(N-1),-1,0,1,(N-1)],shape = (N,N)).toarray()
    DiffMatrix_right_v[0][0]= 1 - sigma_v;DiffMatrix_right_v[0][-1]=0.;DiffMatrix_right_v[-1][0] = 0.;DiffMatrix_right_v[-1][-1] = 1-sigma_v
   
elif BC==3:
    #Dirichlet BC
    print(1)


#-------------------------
# Time stepping
#-------------------------
U = u0
V = v0

Usol = [];#Usol.append(U)
Vsol = [];#Vsol.append(V)



j = 0
for i in range(tsteps):
    if i%int(time_gap/dt) ==0:
        plt.ylim((-1,1.1));plt.xlim((-L/2,L/2))
        plt.xlabel('X');plt.ylabel('Concentration')
        plt.plot(X,U);plt.plot(X,V)
        plt.title('Time = '+str(Time[i]).zfill(5))
        plt.show()
        plt.savefig(filenames[j],dpi = 300)
        j = j+1
        plt.clf()
    Nu = dt*(U-U**3-V)
    Nv = dt*epsilon*(U-b*V+a)
    Unew = np.linalg.solve(DiffMatrix_left_u,DiffMatrix_right_u.dot(U) + Nu)
    Vnew = np.linalg.solve(DiffMatrix_left_v,DiffMatrix_right_v.dot(V) + Nv)
    U = Unew
    V = Vnew
    Usol.append(U)
    Vsol.append(V)
    
#make_animation(filenames,output_path)        

Usol = np.array(Usol)
Vsol = np.array(Vsol)
fig,ax = plt.subplots()
plt.xlabel('x');plt.ylabel('Time')
heatmap = ax.pcolor(X,Time,Usol,vmin = -1,vmax = 1.1)
plt.title('b='+str(b))
plt.savefig('FHN1D_kymograph_b'+str(b),dpi = 300)


    