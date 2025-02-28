## Generate the data for the raster plots for the reliability experiment
# on Luka's bursting circuit.

# Reliability experiments on Luka's circuit
# Why 'new'? I think this is the new protocol, for the revised paper, as
# discussed with Rodolphe.

using Plots, Random, Distributions
using DifferentialEquations, LinearAlgebra, JLD

include("LR_odes.jl")
# Need to put a seed here?

num_trials = 8

max_error = 0.1 # 0.1 gives a mismatch of up to +/- 5%
max_tau_error = 0.04

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

delta_h = -0.5
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


# Now run the observer.

# First, generate the mismatch for it.
mis = (rand(Uniform(0,max_error),4).-max_error/2).+1
mis_tau = (rand(Uniform(0,max_tau_error),2).-max_tau_error/2).+1
delta_ests_raw = [0.01,0.01,-1.5,-1.5]
delta_ests = delta_ests_raw.*mis
tau_ests = [tau_s, tau_us].*mis_tau
println(tau_ests)

global delta_est_values[:,1] = delta_ests
global tau_est_values[:,1] = tau_ests

Iappobs = t -> -2.6 + 0.4*sin(0.001*t)
IappobsStep = t -> -2.2

γ = 2;
α = 0.0001;
# Tfinal= 65000.0; # Non-diag observer converges faster.
Tfinal= 60000.0;
tspan=(0.0,Tfinal);

x0 = [-1.9 -1.9 -1.9];
xh0 = [-1 -1 -1];
θ̂₀ = [.1 .1 .1 .1];
P₀ = [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1]; # For non-diag observer.
# P₀ = [1 1 1 1];
Ψ₀ = [0 0 0 0];
u0 = [x0 xh0 θ̂₀ P₀ Ψ₀];

println("Learning...")
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iappobs,delta_ests,α,γ,tau_ests);
probObs = ODEProblem(LR_observer_noinact_nondiag!,u0,tspan,p) # Simulation without noise (ODE)
solObs = solve(probObs,Euler(),adaptive=false,dt=dt)

p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,IappobsStep,delta_ests,α,γ,tau_ests);
probObsStep = ODEProblem(LR_observer_noinact_nondiag!,u0,tspan,p) # Simulation without noise (ODE)
solObsStep = solve(probObsStep,Euler(),adaptive=false,dt=dt)
println("Finished learning.")

# global plt0 = plot(solObs.t, solObs[1,:])
# global plt1 = plot(solObs.t, solObs[7,:])
# plot!(solObs.t, -solObs[8,:])
# global plt2 = plot(solObs.t, solObs[9,:])

# Truncated figures
j = size(solObs)[3]
# i = round(Int,1*j/5)
# plot(solObs.t[i:j], solObs[7,i:j])
# plot!(solObs.t[i:j], -solObs[8,i:j]) # Plotting the negative so can compare!

# Learned parameters
afnl = mean(solObs[7,j-10000:j]);
aspl = mean(solObs[8,j-10000:j]);
asnl = mean(solObs[9,j-10000:j]);
auspl = mean(solObs[10,j-10000:j]);
global thetalearned = [afnl aspl asnl auspl];

# And for the step
afnlS = mean(solObsStep[7,j-10000:j]);
asplS = mean(solObsStep[8,j-10000:j]);
asnlS = mean(solObsStep[9,j-10000:j]);
ausplS = mean(solObsStep[10,j-10000:j]);
global thetalearnedStep = [afnlS asplS asnlS ausplS];

for idx in 1:num_trials
    println("Trial Number: $idx")
    mis = (rand(Uniform(0,max_error),4).-max_error/2).+1
    mis_tau = (rand(Uniform(0,max_tau_error),2).-max_tau_error/2).+1
    delta_ests = delta_ests_raw.*mis
    tau_ests = [tau_s, tau_us].*mis_tau

    global delta_est_values[:,idx+1] = delta_ests
    global tau_est_values[:,idx+1] = tau_ests


    x0 = [-1.5 -1.5 -1.5]
    u0 = x0
    Tfinal= Tfinalrel
    tspan=(0.0,Tfinal)
    p=(afnl,aspl,asnl,auspl,dfn,dsp,dsn,dusp,tau_ests[1],tau_ests[2],noise,delta_ests)
    probLearned = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
    solLearned = solve(probLearned,Euler(),adaptive=false,dt=dt)
    global Mis[:,idx] = solLearned[1,:]

    # And for the step
    p=(afnlS,asplS,asnlS,ausplS,dfn,dsp,dsn,dusp,tau_ests[1],tau_ests[2],Istep,delta_ests)
    probLearnedStep = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
    solLearnedStep = solve(probLearnedStep,Euler(),adaptive=false,dt=dt)
    global MisStep[:,idx] = solLearnedStep[1,:]

    # p2=plot(solRef.t, solRef[1,:])
    # plot!(solLearned.t, solLearned[1,:])
end

save("sec4_LR_new_burst_withstep.jld","noise",noise,
    "delta_est_values",delta_est_values,"tau_est_values",tau_est_values,
    "thetalearned",thetalearned,"thetalearnedStep",thetalearnedStep,
    "t",t,"Ref",Ref,"Mis",Mis,
    "RefStep",RefStep,"MisStep",MisStep)