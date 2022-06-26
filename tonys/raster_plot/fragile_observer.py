from neuron_model import dyns, neuron, network,e_dyns,exp,neuron_diag
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp,odeint
from typing import List
import pickle
from numba import jit,njit,typeof
from numba.typed import List as NumbaList

num_trials = 1 # of a mismatch neuron.

dyns_array=[e_dyns] 

mis_arr = np.zeros((12,2,num_trials)) # To save later.
mis_t_arr = np.zeros((12,4,num_trials))
for i in range(num_trials):# generate mismatch
    mis_temp=(np.random.rand(12,2)*0.02-0.01)+1.0
    mis_temp2=(np.random.rand(12,4)*0.01-0.005)*2+1.0
    dyns_array.append(dyns(mis_temp,mis_temp2,True,True))

    mis_arr[:,:,i] = mis_temp
    mis_t_arr[:,:,i] = mis_temp2
    
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

# Learn with diagonalised observer
# set simulation time
print("LEARNING")
#TfinalLearn=20000.0
TfinalLearn=15000.0
tspanLearn=[0.0,TfinalLearn]
ts_vec = np.linspace(0,TfinalLearn,int(TfinalLearn/0.1))
lenT = len(ts_vec)
thetalearned = np.zeros((8,2,lenT))
gamma=5
alpha=0.001
variable_mask1=np.array([1.,1.,1.,1.,1.,1.,1.,1.]) # no synape connections so only 8 parameters
for (idx,dyn) in enumerate(dyns_array):
    print("Next Trial")   
    cell1=neuron(NumbaList(
                [gCaT,gKd,gH,gNa,gA,gCaS,gKCa,C,gleak,KdCa,kc] 
            ),
                 e_dyns,dyn,ob_type="V"
            )
    cell1.set_input(NumbaList([2,0,0,0,0,0,0,2,0,100]))
    # cell1.set_input(NumbaList([1.2,0,0,0,0,0,0,2,5,0.01]))
    cell1.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
    cell1.set_tau(tmKCa,1.,10.)
    cell1.set_hyp(gamma,alpha,variable_mask1)
    # get initial condition 
    X0=cell1.init_cond_OB(-60)
    X0[cell1.pos_dinamics+2] = 1 # Initial estimate of gT
    X0[cell1.pos_dinamics+7] = 1 # Initial estimate of gS
    # start simulation and the timer 
    sol=solve_ivp(cell1.OB_ODE_V_equ,tspanLearn , X0,t_eval=ts_vec)#use Ca observer
    print('Estimated value')
    print(np.mean(sol.y[cell1.pos_dinamics+2][-1000:]))
    print(np.mean(sol.y[cell1.pos_dinamics+7][-1000:]))
    plt.figure();plt.plot(sol.t,sol.y[0][:])
    # plt.figure();plt.plot(sol.t, sol.y[cell1.pos_dinamics+2][:]);plt.show()
    print(sol.y[cell1.pos_dinamics+2][:].shape)
    thetalearned[:,idx,:] = sol.y[cell1.pos_dinamics:cell1.pos_Theta]
    # True values 120 0 0.5 0 80 0.3 30 4

np.savez("sec3_fragile_observer.npz",mis_arr=mis_arr,mis_t_arr=mis_t_arr,
                            t=ts_vec,thetalearned=thetalearned)

# plt.show()
