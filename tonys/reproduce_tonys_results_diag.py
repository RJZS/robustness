from neuron_diag_model import dyns, neuron_diag,e_dyns,exp

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp,odeint
from typing import List
import pickle
from numba import jit,njit,typeof
from numba.typed import List as NumbaList

mis=np.array([[0.92609936, 1.04030251],
 [1.08990199, 0.96314537],
 [0.95855323, 1.04500793],
 [0.94757596, 1.07437747],
 [1.04912963, 1.09813075],
 [0.97847925, 0.94515563],
 [1.08024781, 1.02569354],
 [1.02212363, 0.99312849],
 [1.09042846, 1.09455804],
 [1.04422248, 1.05700935],
 [0.91937835, 1.00623917],
 [0.95225629, 1.04815975]])
# mis2=(np.random.rand(12,2)*0.1-0.05)+1.0
mis2=np.array([[0.9522494,  1.00529835],
 [0.98031493, 1.03712259],
 [0.97079993, 1.01578003],
 [0.96851134, 1.01300902],
 [1.03317727, 0.95304496],
 [0.9618265,  1.02377636],
 [0.9781444,  1.00494114],
 [0.96325415, 1.04678504],
 [0.99120024, 0.95379498],
 [1.04983736, 0.96970394],
 [0.96161522, 1.02271901],
 [1.00639862, 1.04303063]])

mis3=(np.random.rand(12,2)*0.02-0.01)+1.0
# mis3=(np.random.rand(12,2)*0.1-0.05)+1.0
# mis3=np.array([[0.99613559, 0.99282902],
#  [0.99108317, 1.00733338],
#  [0.990744,   1.00030067],
#  [1.00442029, 1.00776527],
#  [0.99843125, 0.99669793],
#  [0.99501247, 1.00510612],
#  [1.00414166, 1.00567561],
#  [0.99150875, 1.00000059],
#  [0.99307946, 1.00737742],
#  [1.00023778, 1.0091104 ],
#  [1.00636472, 0.99636941],
#  [1.00368592, 1.00672459]])

mis_t=(np.random.rand(12,4)*0.01-0.005)*2+1.0
# mis_t=(np.random.rand(12,4)*0.04-0.02)*2+1.0
# mis_t=np.array([[1.00624955, 1.00653017, 0.99632439, 1.00371368],
#  [0.99696342, 0.992745,   0.99533094, 0.9936827 ],
#  [1.00657543, 0.99577595, 1.00797668, 1.0082202 ],
#  [0.9968211,  1.00796824, 1.00832337, 0.9922949 ],
#  [1.00940513, 0.99072558, 0.99883522, 0.99657906],
#  [0.99210138, 0.99354303, 1.00782988, 1.00993412],
#  [1.00346248, 1.00776864, 1.0050221,  1.0009537 ],
#  [0.99122363, 1.007003,   1.00363552, 0.99600824],
#  [1.00503467, 0.99437215, 0.99785033, 0.99007775],
#  [1.00216428, 0.99226534, 1.00489528, 0.99316654],
#  [1.00536349, 0.99371844, 0.99261024, 0.99874712],
#  [0.99540985, 1.00244202, 1.00577586, 1.00434942]])

# generate internal dynamics with mismatches for robustness simulation
dyns_time=dyns(mis2,mis_t,True,False)
dyns_act=dyns(mis2,mis_t,False,True)
dyns_time_act=dyns(mis2,mis_t,True,True)

dyns_time2=dyns(mis3,mis_t,True,False)
dyns_act2=dyns(mis3,mis_t,False,True)
dyns_time_act2=dyns(mis3,mis_t,True,True)

ones=np.ones 
# Model parameters (global)
VNa = 50
VK = -80
VCa = 80
VH = -20
Vleak = -50
VSyn = -75
taus = 10.
C = 1.
taumean=30.

# Model parameters (mean)
Iappvec = 0.*ones(5)

Iappvec = 0.*ones(5) + 0/2*(np.random.rand(5)-0.5)
kcvec = 0.94*ones(5)
KdCavec = 3.*ones(5)

gleakvec = 0.01*ones(5)

gNavec = [652.814,503.58,634.723,459.807,616.433]
gCaTvec = [0.992884,0.824788,1.04958,1.14911,0.971846]
gCaTvec_MOD = [2.38666,2.19092,2.3409,2.82957,3.52574]
gCaSvec = [3.51903,3.55081,3.36012,2.49984,2.50894]
gCaSvec_MOD = [8.52841,8.00527,9.24152,9.73135,8.79102]
gAvec = [51.5008,58.3269,40.3572,58.5102,51.8502]
gKdvec = [109.659,84.1655,74.7833,91.1938,111.87]
gKCavec = [63.0373,54.1351,65.7822,55.9324,70.2168]
gHvec = [0.107445,0.0929811,0.078182,0.083414,0.0887343]
tmKCavec = [3.93883,3.24514,5.55055,12.6351,16.6223]

gsyn12 = 0.07635083670743605
gsyn13 = 0.07988922275521991
gsyn21 = 0.07716951193265496
gsyn45 = 0.07762158408977728
gsyn53 = .08835472319081544
gsyn54 = 0.11712333284581126
gEl23 = 0.021083916923133217
gEl43 = 0.022874860313218278

@jit
def sigmoid(x,tau):
    return 1/(1+exp(-tau*x))

# Noise Generation
noise=(np.random.normal(size=200000))*10 # sf for bursting
noise[0]=0
for i in range(len(noise)-1):
    noise[i+1]=noise[i]+(noise[i+1]-noise[i])/1000

noise=(noise-noise.mean())*2
plt.plot(noise[0:10000])
print(noise.var())
np.save('noise_diag_model.npy',noise)

# noise = np.load('noise_orig_model.npy')

## Parameters for Bursting
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

cell1d=neuron_diag(NumbaList(
            [gCaT,gKd,gH,gNa,gA,gCaS,gKCa,C,gleak,KdCa,kc]
        ),
             e_dyns,e_dyns,ob_type='Ca'
        )
cell1d.set_input(NumbaList([1.2,0,0,0,0,0,0,2,0,100]),0.000)
cell1d.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
cell1d.set_tau(tmKCavec[0],1.,10.)

def noisy_input_neuron(t,u):
    return cell1d.equ_noise(t,u,noise[int(t*10)-1])

X0=cell1d.init_cond(-0)
# set simulation time
Tfinal=10000.0
tspan=[0.0,Tfinal]
T=np.linspace(0., Tfinal, 100000)
# start simulation and the timer 
start = time.time()
solBurstRef=odeint(noisy_input_neuron, X0,T,tfirst=True)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

plt.plot(T,solBurstRef[:,0])

# Now the mismatched neuron
cell2d=neuron_diag(NumbaList(
            [gCaT,gKd,gH,gNa,gA,gCaS,gKCa,C,gleak,KdCa,kc]
        ),
             dyns_time_act2,dyns_time_act2,ob_type='Ca'
        )
cell2d.set_input(NumbaList([1.2,0,0,0,0,0,0,2,0,100]),0.000)
cell2d.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
cell2d.set_tau(tmKCavec[0],1.,10.)

def noisy_input_neuron2(t,u):
    return cell2d.equ_noise(t,u,noise[int(t*10)-1])

# get initial condition 
X0=cell2d.init_cond(-5) # changed initial conds

# start simulation and the timer 
start = time.time()
solBurstMis=odeint(noisy_input_neuron2, X0,T,tfirst=True)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

plt.plot(T,solBurstMis[:,0])

plt.plot(T,solBurstRef[:,0],T,0.8*solBurstMis[:,0])

# Now try learning the mismatched neuron
cell3d=neuron_diag(NumbaList(
            [gCaT,gKd,gH,gNa,gA,gCaS,gKCa,C,gleak,KdCa,kc]
        ),
             e_dyns,dyns_time_act2,ob_type='Ca'
        )
cell3d.set_input(NumbaList([2,0,0,0,0,0,0,2,0,100]),0.000)
cell3d.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
cell3d.set_tau(tmKCavec[0],1.,10.)
gamma=2.
alpha=0.001
variable_mask1=np.array([0.,0.,1.,0.,0.,0.,0.,1.]) # no synape connections so only 8 parameters
cell3d.set_hyp(gamma,alpha,variable_mask1)
# get initial condition 
X0=cell3d.init_cond_OB(-60)

# Make the initial condition easier
X0[cell3d.pos_dinamics+2] = 1 # Initial estimate of gT
X0[cell3d.pos_dinamics+7] = 1 # Initial estimate of gS

# set simulation time
Tfinal=60000.0
tspan=[0.0,Tfinal]
# start simulation and the timer 
start = time.time()
solBurstCaObs=solve_ivp(cell3d.OB_ODE_Ca_equ,tspan , X0)#use Ca observer
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# Convergence of theta_hat
idx_tmp = -200000
plt.plot(solBurstCaObs.t[idx_tmp:], solBurstCaObs.y[cell3d.pos_dinamics+2][idx_tmp:])
plt.plot(solBurstCaObs.t[idx_tmp:], solBurstCaObs.y[cell3d.pos_dinamics+7][idx_tmp:])

# Learned parameters
gTl = np.mean(solBurstCaObs.y[cell3d.pos_dinamics+2][-1000:])
gSl = np.mean(solBurstCaObs.y[cell3d.pos_dinamics+7][-1000:])

# Now simulate learned mismatch neuron.
cell4d=neuron_diag(NumbaList(
            [gTl,gKd,gH,gNa,gA,gSl,gKCa,C,gleak,KdCa,kc]
        ),
             dyns_time_act2,dyns_time_act2,ob_type='Ca'
        )
cell4d.set_input(NumbaList([1.2,0,0,0,0,0,0,2,0,100]),0.000)
cell4d.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
cell4d.set_tau(tmKCavec[0],1.,10.)

def noisy_input_neuron4(t,u):
    return cell4d.equ_noise(t,u,noise[int(t*10)-1])

# get initial condition 
X0=cell4d.init_cond(-5)
# set simulation time
Tfinal=10000.0
tspan=[0.0,Tfinal]
T=np.linspace(0., Tfinal, 100000)
# start simulation and the timer 
start = time.time()
solBurstLearned=odeint(noisy_input_neuron4, X0,T,tfirst=True)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

plt.plot(T,solBurstLearned[:,0])

plt.plot(T,solBurstRef[:,0],T,0.8*solBurstLearned[:,0])

## VOLTAGE OBSERVER
## Now try learning with voltage instead of calcium
cell5=neuron_diag(NumbaList(
            [gCaT,gKd,gH,gNa,gA,gCaS,gKCa,C,gleak,KdCa,kc]
        ),
             e_dyns,dyns_time_act2,ob_type='V'
        )
cell5.set_input(NumbaList([2,0,0,0,0,0,0,2,0,100]),0.000)
cell5.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
cell5.set_tau(tmKCavec[0],1.,10.)
gamma=10.
alpha=0.001
variable_mask2=np.array([1.,1.,1.,1.,1.,0.,1.,1.]) # no synape connections so only 8 parameters
cell5.set_hyp(gamma,alpha,variable_mask2)
# get initial condition 
X0=cell5.init_cond_OB(-60)

# set simulation time
Tfinal=20000.0
tspan=[0.0,Tfinal]
# start simulation and the timer 
start = time.time()
solBurstVObs=solve_ivp(cell5.OB_ODE_V_equ,tspan , X0)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# Learned parameters
gNalV = np.mean(solBurstVObs.y[cell5.pos_dinamics][-1000:])
gHlV = np.mean(solBurstVObs.y[cell5.pos_dinamics+1][-1000:])
gTlV = np.mean(solBurstVObs.y[cell5.pos_dinamics+2][-1000:])
gAlV = np.mean(solBurstVObs.y[cell5.pos_dinamics+3][-1000:])
gKdlV = np.mean(solBurstVObs.y[cell5.pos_dinamics+4][-1000:])
# gleaklV = np.mean(solBurstVObs.y[cell5.pos_dinamics+5][-1000:])
gKCalV = np.mean(solBurstVObs.y[cell5.pos_dinamics+6][-1000:])
gSlV = np.mean(solBurstVObs.y[cell5.pos_dinamics+7][-1000:])

# Now simulate learned mismatch neuron.
cell6=neuron_diag(NumbaList(
            [gTlV,gKdlV,gHlV,gNalV,gAlV,gSlV,gKCalV,C,gleak,KdCa,kc]
        ),
             dyns_time_act2,dyns_time_act2,ob_type='V'
        )
cell6.set_input(NumbaList([1.2,0,0,0,0,0,0,2,0,100]),0.000)
cell6.set_rev(NumbaList([VNa,VCa,VK,VH,Vleak,VSyn]))
cell6.set_tau(tmKCavec[0],1.,10.)

def noisy_input_neuron6(t,u):
    return cell6.equ_noise(t,u,noise[int(t*10)-1])
    
# get initial condition 
X0=cell6.init_cond(-5)
# set simulation time
Tfinal=10000.0
tspan=[0.0,Tfinal]
T=np.linspace(0., Tfinal, 100000)
# start simulation and the timer 
start = time.time()
solBurstVLearned=odeint(noisy_input_neuron6, X0,T,tfirst=True)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

plt.plot(T,solBurstVLearned[:,0])