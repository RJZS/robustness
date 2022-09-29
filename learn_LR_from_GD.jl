## Generate the data for the raster plots for the reliability experiment
# on Luka's bursting circuit.

# Training data is generated from a conductance-based model.
# Derived from 'LR_relfigs_Ca_burst_gen.jl'.

using Plots, Random, Distributions
using DifferentialEquations, LinearAlgebra, JLD

include("conductance_based/GD_odes.jl")

## Constant simulation parameters

## Definition of reversal potential values. 
const VNa = 40.; # Sodium reversal potential
const VK = -90.; # Potassium reversal potential
const VCa = 120.; # Calcium reversal potential
const VH= -40.; # Reversal potential for the H-current (permeable to both sodium and potassium ions)
const Vl = -50.; # Reversal potential of leak channels

const C=0.1; # Membrane capacitance
const αCa=0.1; # Calcium dynamics (L-current)
const βCa=0.1 # Calcium dynamics (T-current)

gl=0.3; # Leak current maximal conductance
gNa=100.; # Sodium current maximal conductance
gKd=65.; # Delayed-rectifier potassium current maximal conductance
gAf=0.; # Fast A-type potassium current maximal conductance
gAs=0.; # Slow A-type potassium current maximal conductance
gKCa=8.; # Calcium-activated potassium current maximal conductance
gCaL=4.; # L-type calcium current maximal conductance
gCaT=0.; # T-type calcium current maximal conductance
gH=0.; # H-current maximal conductance

# Initial conditions
x0 = init_neur(-70.);

Tfinal=6000.0
tspan=(0.0,Tfinal)

Iapp = 4.
# Current pulses
I1= 0. # Amplitude of first pulse
ti1=2000 # Starting time of first pulse
tf1=3000 # Ending time of first pulse
I2=-0. # Amplitude of second pulse
ti2=000. # Starting time of second pulse
tf2=0000. # Ending time of first pulse

## Current-clamp experiment
# Parameter vector for simulations
p=(Iapp,I1,I2,ti1,tf1,ti2,tf2,gNa,gKd,gAf,gAs,gKCa,gCaL,gCaT,gH,gl)

# Simulation
# Using the calcium observer
prob = ODEProblem(CBM_ODE_withLR,x0,tspan,p); # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1);
plot(sol.t, sol[1,:])

### Now use that to train Luka's.

include("LR_odes.jl")
# Need to put a seed here?

num_trials = 8

max_error = 0.1 # 0.1 gives a mismatch of up to +/- 5%
max_tau_error = 0.04 # Try raising this.
max_alpha_error = 0.2 # Mismatch on afn and asp in the reliability experiments.

d = Normal(0,1)
noise_sf = 12

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = -2
asp = 2
asn =  -1.5
ausp =  2

# NOTE: These are not directly used by LR_ODE_rel!
# If you change these, remember to change 'delta_ests'.
dfn = 0
dsp = 0
dsn = -1.5
dusp = -1.5

# If you change this, make sure to also change 'delta_ests'
delta_ests_true = [dfn,dsp,dsn,dusp]

beta = 2


Tfinalrel = 20000
dt = 0.1

# Noise-generated current
noise = rand(d, round(Int, Tfinalrel/dt+1))*noise_sf
Iconst = -2.6

for i in eachindex(noise)
    i == 1 ? noise[i] = 0 : noise[i]=noise[i-1]+(noise[i]-noise[i-1])/2000
end
noise = noise .+ Iconst

# Initialise arrays which will later be saved as .jld files.
delta_est_values = zeros(4,num_trials+1) # '+1' is for the observer.
tau_est_values = zeros(2,num_trials+1)
alpha_est_values = zeros(2,num_trials)

Ref = zeros(Int(Tfinalrel/dt+2))
Mis = zeros(Int(Tfinalrel/dt+2),num_trials)

RefStep = zeros(Int(Tfinalrel/dt+2))
MisStep = zeros(Int(Tfinalrel/dt+2),num_trials)

thetalearned = zeros(4)
thetalearnedStep = zeros(4)

# Run the reference simulation.
Tfinal= Tfinalrel
tspan=(0.0,Tfinal)
x0 = [-1.8 -1.8 -1.8]
u0 = x0
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,noise,delta_ests_true)
probRef = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solRef = solve(probRef,Euler(),adaptive=false,dt=dt)
t = solRef.t
Ref = solRef[1,:]

# Now the step version
Istep = -2.2*ones(Int(Tfinalrel/dt+1))
Istep[1:30000] .= -2.6
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Istep,delta_ests_true)
probRefStep = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solRefStep = solve(probRefStep,Euler(),adaptive=false,dt=dt)
RefStep = solRefStep[1,:]


# Now run the observer. NOTE: using two alphas in the P dynamics.

# First, generate the mismatch for it.
mis = (rand(Uniform(0,max_error),4).-max_error/2).+1
mis_tau = (rand(Uniform(0,max_tau_error),2).-max_tau_error/2).+1
delta_ests_raw = [0.01,0.01,-1.5,-1.5]
delta_ests = delta_ests_raw.*mis
tau_ests = [tau_s, tau_us].*mis_tau

global delta_est_values[:,1] = delta_ests
global tau_est_values[:,1] = tau_ests

Iappobs = t -> -2.6 + 0.4*sin(0.001*t)
IappobsStep = t -> -2.2

γ = 0.1;
# α = 0.0001;
α = 0.01;
# Tfinal= 65000.0; # Non-diag observer converges faster.
Tfinal= 60000.0;
tspan=(0.0,Tfinal);

x0 = [-1.9 -1.9 -1.9 -1.5];
xh0 = [-1 -1 -1 -1];
θ̂₀ = [.1 .1];
P₀ = [1 0 0 1];
Ψ₀ = [0 0];
u0 = [x0 xh0 θ̂₀ P₀ Ψ₀];

println("Learning...")
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iappobs,delta_ests,α,γ,tau_ests);
# Below is mismatch-free version for testing.
# p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iappobs,delta_ests_true,α,γ,[tau_s, tau_us]);
probObs = ODEProblem(LR_Ca_observer_noinact_nondiag!,u0,tspan,p) # Simulation without noise (ODE)
solObs = solve(probObs,Euler(),adaptive=false,dt=dt)

tspanStep = (0.0,1.0); # Skip the step simulation for now.
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,IappobsStep,delta_ests,α,γ,tau_ests);
probObsStep = ODEProblem(LR_Ca_observer_noinact_nondiag!,u0,tspanStep,p) # Simulation without noise (ODE)
solObsStep = solve(probObsStep,Euler(),adaptive=false,dt=dt)
println("Finished learning.")

plt0 = plot(solObs.t, solObs[1,:]) # V and 'Ca'.
plot!(solObs.t, solObs[4,:], linewidth=2)

plt1 = plot(solObs.t, solObs[4,:], linewidth=2) # Ca and Cah.
plot!(solObs.t, solObs[8,:], linewidth=2)

# Theta hat plots.
plt2 = plot(solObs.t, solObs[9,:], linewidth=2)
plt3 = plot(solObs.t, solObs[10,:], linewidth=2)

# Truncated figures
j = size(solObs)[3]
# i = round(Int,1*j/5)
# plot(solObs.t[i:j], solObs[7,i:j])
# plot!(solObs.t[i:j], -solObs[8,i:j]) # Plotting the negative so can compare!

# Learned parameters
afnl = mean(solObs[7,j-100000:j]);
aspl = mean(solObs[8,j-100000:j]);
asnl = mean(solObs[9,j-100000:j]);
auspl = mean(solObs[10,j-100000:j]);
global thetalearned = [afnl aspl asnl auspl];

# And for the step
afnlS = mean(solObsStep[7,1:5]); # 5 just because I'm not running the step.
asplS = mean(solObsStep[8,1:5]);
asnlS = mean(solObsStep[9,1:5]);
ausplS = mean(solObsStep[10,1:5]);
global thetalearnedStep = [afnlS asplS asnlS ausplS];

for idx in 1:num_trials
    println("Trial Number: $idx")
    mis = (rand(Uniform(0,max_error),4).-max_error/2).+1
    mis_tau = (rand(Uniform(0,max_tau_error),2).-max_tau_error/2).+1
    delta_ests = delta_ests_raw.*mis
    tau_ests = [tau_s, tau_us].*mis_tau
    
    mis_alpha = (rand(Uniform(0,max_alpha_error),2).-max_alpha_error/2).+1
    alpha_ests = [afn, asp].*mis_alpha

    global delta_est_values[:,idx+1] = delta_ests
    global tau_est_values[:,idx+1] = tau_ests
    global alpha_est_values[:,idx] = alpha_ests

    x0 = [-1.5 -1.5 -1.5]
    u0 = x0
    Tfinal= Tfinalrel
    tspan=(0.0,Tfinal)
    p=(alpha_ests[1],alpha_ests[2],asnl,auspl,dfn,dsp,dsn,dusp,tau_ests[1],tau_ests[2],noise,delta_ests)
    probLearned = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
    solLearned = solve(probLearned,Euler(),adaptive=false,dt=dt)
    global Mis[:,idx] = solLearned[1,:]

    # And for the step
    p=(alpha_ests[1],alpha_ests[2],asnlS,ausplS,dfn,dsp,dsn,dusp,tau_ests[1],tau_ests[2],Istep,delta_ests)
    probLearnedStep = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
    solLearnedStep = solve(probLearnedStep,Euler(),adaptive=false,dt=dt)
    global MisStep[:,idx] = solLearnedStep[1,:]

    # p2=plot(solRef.t, solRef[1,:])
    # plot!(solLearned.t, solLearned[1,:])
end

# Ref vs a Mismatch
pltrun = plot(solRef.t, solRef[1,:])
plot!(solRef.t, Mis[:,1]) # Second run.


save("sec4_LR_GDLR_burst_withstep.jld","noise",noise,
    "delta_est_values",delta_est_values,"tau_est_values",tau_est_values,
    "alpha_est_values",alpha_est_values,
    "thetalearned",thetalearned,"thetalearnedStep",thetalearnedStep,
    "t",t,"Ref",Ref,"Mis",Mis,
    "RefStep",RefStep,"MisStep",MisStep)