from neuron_model import dyns, neuron, network,e_dyns,exp,neuron_diag
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp,odeint
from typing import List
import pickle
from numba import jit,njit,typeof
from numba.typed import List as NumbaList

num_trials = 2 # of a mismatch neuron.

dyns_array=[e_dyns] 

mis_arr = np.zeros((12,2,num_trials)) # To save later.
mis_t_arr = np.zeros((12,4,num_trials))
for i in range(num_trials):# generate mismatch
    mis_temp=(np.random.rand(12,2)*0.1-0.05)+1.0
    mis_temp2=(np.random.rand(12,4)*0.01-0.005)*2+1.0
    dyns_array.append(dyns(mis_temp,mis_temp2,True,True))

    mis_arr[:,:,i] = mis_temp
    mis_t_arr[:,:,i] = mis_temp2
    
noise=(np.random.normal(size=200000))*60 #generate noise
noise[0]=0
for i in range(len(noise)-1): 
    noise[i+1]=noise[i]*0.9999+noise[i+1]*(1-0.9999)
    
noise=(noise-noise.mean())
plt.plot(noise[0:100000])
print(noise.var())

# Model parameters (global)
VNa = 40
VK = -90
VCa = 120
VH = -40
Vleak = -50
VSyn = -75
taus = 10.
C = 0.1
taumean=30.

# Model parameters (mean)
Iapp = 0.
kc = 0.94
KdCa = 3.

gleak = 0.3

gNa = 120
gCaT= 0.5
gCaS = 4
gA = 0
gKd = 80
gKCa = 30
gH = 0
tmKCa = 3.93883

# set simulation time
Tfinal=10000.0
tspan=[0.0,Tfinal]
T=np.linspace(0., Tfinal, 100000)

# init params for saving
lenT = len(T)
t = np.zeros(lenT); Rel = np.zeros(lenT); Mis = np.zeros((lenT,num_trials))
Learned = np.zeros((lenT,num_trials))
# Pre-learning simulations.
print("PRE LEARNING")
sol_array=[]
for (idx, dyn) in enumerate(dyns_array):
    print("Trial: {}".format(idx))
    cell1=neuron(NumbaList(
            [gCaT,gKd,gH,gNa,gA,gCaS,gKCa,C,gleak,KdCa,kc]
        ),
             dyn,dyn
        )
    cell1.set_input(NumbaList([1.2,0,0,0,0,0,0,2,0,100]),0.000)
    cell1.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
    cell1.set_tau(tmKCa,1.,10.)

    def noisy_input_neuron(t,u):
        return cell1.equ_noise(t,u,noise[int(t*10)-1])

    # get initial condition 
    X0=cell1.init_cond(-0)
    # start simulation
    sol=solve_ivp(noisy_input_neuron, tspan,X0,t_eval=T)
    sol_array.append([sol.t,sol.y[0]])
    if idx == 1:
        t = sol.t
        Ref = sol.y[0]
    else:
        Mis[:,idx-1] = sol.y[0]
    
# Learn with diagonalised observer
# set simulation time
print("LEARNING")
TfinalLearn=20000.0
tspanLearn=[0.0,TfinalLearn]
sol_OB_Par_array2=[]#learned parameters
gamma=10.
alpha=0.001
variable_mask1=np.array([0.,0.,1.,0.,0.,0.,0.,1.]) # no synape connections so only 8 parameters
for dyn in dyns_array:
    print("Next Trial")   
    cell1=neuron_diag(NumbaList(
                [gCaT,gKd,gH,gNa,gA,gCaS,gKCa,C,gleak,KdCa,kc] 
            ),
                 e_dyns,dyn,
            )
    cell1.set_input(NumbaList([2,0,0,0,0,0,0,2,0,100]))
    cell1.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
    cell1.set_tau(tmKCa,1.,10.)
    cell1.set_hyp(gamma,alpha,variable_mask1)
    # get initial condition 
    X0=cell1.init_cond_OB(-60)
    # start simulation and the timer 
    sol=solve_ivp(cell1.OB_ODE_equ,tspanLearn , X0)#use Ca observer
    print('Estimated value')
    print(np.mean(sol.y[cell1.pos_dinamics+2][-1000:]))
    print(np.mean(sol.y[cell1.pos_dinamics+7][-1000:]))
    sol_OB_Par_array2.append([np.mean(sol.y[cell1.pos_dinamics+2][-1000:]),np.mean(sol.y[cell1.pos_dinamics+7][-1000:])])

print("FINISHED LEARNING")
sol_OB_array2=[]# Testing
for i in range(len(sol_OB_Par_array2)):
    print("Trial: {}".format(i))
    cell1=neuron(NumbaList(
            [sol_OB_Par_array2[i][0],gKd,gH,gNa,gA,sol_OB_Par_array2[i][1],gKCa,C,gleak,KdCa,kc]
        ),
             dyns_array[i],dyns_array[i]
        )
    cell1.set_input(NumbaList([1.2,0,0,0,0,0,0,2,0,100]),0.000)
    cell1.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
    cell1.set_tau(tmKCa,1.,10.)

    def noisy_input_neuron(t,u):
        return cell1.equ_noise(t,u,noise[int(t*10)-1])

    # get initial condition 
    X0=cell1.init_cond(-0)

    # start simulation and the timer 
    sol=solve_ivp(noisy_input_neuron, tspan,X0,t_eval=T)
    sol_OB_array2.append([sol.t,sol.y[0]])
    if i > 1: # Skip the first one, as that's the ref neuron.
        Learned[:,i-1] = sol.y[0]

np.savez("sec4_CB_burst.npz",noise=noise,mis_arr=mis_arr,mis_t_arr=mis_t_arr,
                            t=t,Rel=Rel,Mis=Mis,Learned=Learned,thetalearned=sol_OB_Par_array2)
