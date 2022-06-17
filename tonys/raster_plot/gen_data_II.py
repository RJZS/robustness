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

dyns_array2=[e_dyns] 

mis_arr = np.zeros((12,2,num_trials)) # To save later.
mis_t_arr = np.zeros((12,4,num_trials))
for i in range(num_trials):
    mis_temp=(np.random.rand(12,2)*0.02-0.01)+1.0
    mis_temp2=(np.random.rand(12,4)*0.01-0.005)*2+1.0
    dyns_array2.append(dyns(mis_temp,mis_temp2,True,True))

    mis_arr[:,:,i] = mis_temp
    mis_t_arr[:,:,i] = mis_temp2
    
noise2=(np.random.normal(size=200000))*60 #generate noise
noise2[0]=0
for i in range(len(noise2)-1):
    noise2[i+1]=noise2[i]*0.9999+noise2[i+1]*(1-0.9999)
    
noise2=(noise2-noise2.mean())
plt.plot(noise2[0:100000])
print(noise2.var())

VNa = 50
VK = -80
VCa = 80
VH = -20
Vleak = -50
VSyn = -75
taus = 10.
C = 1.

gleak = 0.01
gNa = 600.
gCaT_gastr = 3.
gCaS_gastr = 8.
gA_gastr = 50.
gKd = 90.
gKCa_gastr = 60.
gH = 0.1
Iapp = 0
kc = 0.94
KdCa = 3.
tmKCa = 2.
tmKCavec = [3.93883,3.24514,5.55055,12.6351,16.6223]

def inti_cond_net_sys(net1):
    result=[]
    for cell in net1.cells:
        result=[*result,*cell.init_cond(-60+(np.random.rand()-0.5)*10)]
    return result

# set simulation time
Tfinal=15000.0
tspan=[0.0,Tfinal]
T=np.linspace(0., Tfinal, 150000)

# init params for saving
lenT = len(T)
t = np.zeros(lenT); Ref = np.zeros((2,lenT)); Mis = np.zeros((2,lenT,num_trials))
Learned = np.zeros((2,lenT,num_trials))
# Pre-learning simulations.
print("PRE LEARNING")
sol_array2=[]
for (idx, dyn) in enumerate(dyns_array2):
    print("Trial: {}".format(idx))
    cells=[]
    Ivec=[-1.4,-1.3]
    for i in range(2):
            cells.append(neuron(NumbaList(
                [gCaT_gastr,gKd,gH,gNa,gA_gastr,gCaS_gastr,gKCa_gastr,C,gleak,KdCa,kc]
            ),
                 dyn,dyn,ob_type='V'
            ))# initialised cells in the STG network
            cells[i].set_input(NumbaList([Ivec[i],0,0,0,0,0,0,2,0,0]))
            cells[i].set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
            cells[i].set_tau(tmKCavec[i],1.,10.)

    # defined network topology (two types of connections, each defined by a matrix)
    net1=network(cells,[[0,0.8],[0.8,0]],[[0.,0.],[0.,0.]])
    
    def noisy_input_net(t,u):
        return net1.sys_equ_noise(t,u,[noise2[int(t*10)-1],0])   
    
    # get initial condition 
    X0=inti_cond_net_sys(net1)

    # start simulation and the timer 
    sol=solve_ivp(noisy_input_net,tspan,X0,t_eval=T)
    sol_array2.append([sol.t,sol.y[0],sol.y[16]])
    if idx == 0:
        t = sol.t
        Ref = [sol.y[0],sol.y[16]]
    else:
        Mis[:,:,idx-1] = [sol.y[0],sol.y[16]]
    
# Learn with diagonalised observer
# set simulation time
print("LEARNING")
TfinalLearn=10000.0
tspanLearn=[0.0,TfinalLearn]
Tlearn=np.linspace(0., TfinalLearn, 100000)

sol_OB_Par_HCO_array2=[] #learned parameters
Ivec=[-1.4,-1.3]
# set hyper parameters for adaptive observer
gamma=10.
alpha=0.001
variable_mask1=np.array([0.,0.,1.,0.,0.,0.,0.,1.,0.])
for dyn in dyns_array2:
    cells=[]
    for i in range(2):
            cells.append(neuron_diag(NumbaList(
                [gCaT_gastr,gKd,gH,gNa,gA_gastr,gCaS_gastr,gKCa_gastr,C,gleak,KdCa,kc]
            ),
                 e_dyns,dyn,
            ))# initialised cells in the STG network
            cells[i].set_input(NumbaList([Ivec[i],0,0,0,0,0,0,2,0,0]))
            cells[i].set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
            cells[i].set_tau(tmKCavec[i],1.,10.)
    cells[0].set_input(NumbaList([Ivec[0],0,0,0,0,0,0,2,5,0.01]))
    cells[0].set_hyp(gamma,alpha,variable_mask1)
    cells[1].set_hyp(gamma,alpha,variable_mask1) 
    # defined network topology (two types of connections, each defined by a matrix)
    net1=network(cells,[[0,0.8],[0.8,0]],[[0.,0.],[0.,0.]])
    net1.cells[0].set_hyp(gamma,alpha,variable_mask1)
    net1.cells[1].set_hyp(gamma,alpha,variable_mask1) 
    # get initial condition 
    X0=[*net1.cells[0].init_cond_OB(-60+5),*net1.cells[1].init_cond_OB(-60-5)]
    
    # start simulation
    sol=solve_ivp(net1.ob_equ,tspan , X0)

    print("Neur 2")
    net1.cells[0].set_input(NumbaList([Ivec[0],0,0,0,0,0,0,2,0,0]))
    net1.cells[1].set_input(NumbaList([Ivec[1],0,0,0,0,0,0,2,5,0.01]))

    # get initial condition 
    X0=[*net1.cells[0].init_cond_OB(-60+5),*net1.cells[1].init_cond_OB(-60-5)]
    
    # start simulation and the timer 
    start = time.time()
    sol_=solve_ivp(net1.ob_equ,tspan , X0)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    print('Estimated value1')
    print(np.mean(sol.y[net1.cells[0].pos_dinamics+2][-10000:]))
    print(np.mean(sol.y[net1.cells[0].pos_dinamics+7][-10000:]))
    

    print('Estimated value2')
    print(np.mean(sol_.y[net1.cells[0].pos_u_sys+net1.cells[1].pos_dinamics+2][-10000:]))
    print(np.mean(sol_.y[net1.cells[0].pos_u_sys+net1.cells[1].pos_dinamics+7][-10000:]))

    sol_OB_Par_HCO_array2.append([np.mean(sol.y[net1.cells[0].pos_dinamics+2][-10000:])
                                         ,np.mean(sol.y[net1.cells[0].pos_dinamics+7][-10000:])
                                        ,np.mean(sol_.y[net1.cells[0].pos_u_sys+net1.cells[1].pos_dinamics+2][-10000:])
                             ,np.mean(sol_.y[net1.cells[0].pos_u_sys+net1.cells[1].pos_dinamics+7][-10000:])
                                ])

print("FINISHED LEARNING")
sol_OB_HCO_array2=[] # Testing
Ivec=[-1.4,-1.3]
for a in range(len(sol_OB_Par_HCO_array2)):
    print("Trial: {}".format(a))
    cells=[]
   
    Svec=[sol_OB_Par_HCO_array2[a][1],sol_OB_Par_HCO_array2[a][3]]
    Tvec=[sol_OB_Par_HCO_array2[a][0],sol_OB_Par_HCO_array2[a][2]]
    for i in range(2):
            cells.append(neuron(NumbaList(
                [Tvec[i],gKd,gH,gNa,gA_gastr,Svec[i],gKCa_gastr,C,gleak,KdCa,kc]
            ),
                 dyns_array2[a],dyns_array2[a],ob_type='V'
            ))# initialised cells in the STG network
            cells[i].set_input(NumbaList([Ivec[i],0,0,0,0,0,0,2,0,0]))
            cells[i].set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
            cells[i].set_tau(tmKCavec[i],1.,10.)



    # defined network topology (two types of connections, each defined by a matrix)
    net1=network(cells,[[0,0.8],[0.8,0]],[[0.,0.],[0.,0.]])
    
    def noisy_input_net(t,u):
        return net1.sys_equ_noise(t,u,[noise2[int(t*10)-1],0])   
    
    # get initial condition 
    X0=inti_cond_net_sys(net1)
   
    # start simulation
    sol=solve_ivp(noisy_input_net,tspan,X0,t_eval=T)
    sol_OB_HCO_array2.append([sol.t,sol.y[0],sol.y[16]])
    if a > 0: # Skip the first one, as that's the ref neuron.
        Learned[:,:,a-1] = [sol.y[0],sol.y[16]]

np.savez("sec4_CB_burst.npz",noise=noise2,mis_arr=mis_arr,mis_t_arr=mis_t_arr,
                            t=t,Ref=Ref,Mis=Mis,Learned=Learned,thetalearned=sol_OB_Par_HCO_array2)
