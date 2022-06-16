## Generate the data for the raster plots for the reliability experiment
# on Luka's bursting circuit.

# Reliability experiments on Luka's circuit

using Plots, Random, Distributions
using DifferentialEquations, LinearAlgebra, JLD

include("LR_odes.jl")
# Need to put a seed here?

num_trials = 20

max_error = 0.1

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
delta_est_values = zeros(4,num_trials)

Ref = zeros(Int(Tfinalrel/dt+2))
Mis = zeros(Int(Tfinalrel/dt+2),num_trials)
Learned = zeros(Int(Tfinalrel/dt+2),num_trials)

thetalearned = zeros(4,num_trials)

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

for idx in 1:num_trials
    println("Trial Number: $idx")
    mis = (rand(Uniform(0,max_error),4).-max_error/2).+1
    delta_ests = [0.01,0.01,-1.5,-1.5].*mis
    global delta_est_values[:,idx] = delta_ests

    # Simulate the mismatch neuron, before learning.
    Tfinal= Tfinalrel
    tspan=(0.0,Tfinal)
    x0 = [-1.5 -1.5 -1.5]
    u0 = x0
    p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,noise,delta_ests)
    probMis = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
    solMis = solve(probMis,Euler(),adaptive=false,dt=dt)
    global Mis[:,idx] = solMis[1,:]

    # p1=plot(solRef.t, solRef[1,:])
    # plot!(solRef.t, solMis[1,:])


    # Now run the observer.
    Iappobs = -2;

    γ = 2;
    α = 0.0001;
    Tfinal= 60000.0;
    tspan=(0.0,Tfinal);

    x0 = [-1.9 -1.9 -1.9];
    xh0 = [-1 -1 -1];
    θ̂₀ = [.1 .1 .1 .1];
    P₀ = [1 1 1 1];
    Ψ₀ = [0 0 0 0];
    u0 = [x0 xh0 θ̂₀ P₀ Ψ₀];

    println("Learning...")
    p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iappobs,delta_ests,α,γ);
    probObs = ODEProblem(LR_observer_noinact!,u0,tspan,p) # Simulation without noise (ODE)
    solObs = solve(probObs,Euler(),adaptive=false,dt=dt)
    println("Finished learning.")

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
    global thetalearned[:,idx] = [afnl aspl asnl auspl];


    # Finally, apply the learned parameters!
    x0 = [-1.5 -1.5 -1.5]
    u0 = x0
    Tfinal= Tfinalrel
    tspan=(0.0,Tfinal)
    p=(afnl,aspl,asnl,auspl,dfn,dsp,dsn,dusp,tau_s,tau_us,noise,delta_ests)
    probLearned = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
    solLearned = solve(probLearned,Euler(),adaptive=false,dt=dt)
    global Learned[:,idx] = solLearned[1,:]

    # p2=plot(solRef.t, solRef[1,:])
    # plot!(solLearned.t, solLearned[1,:])
end

save("sec4_LR_burst.jld","noise",noise,
    "delta_est_values",delta_est_values,"thetalearned",thetalearned,
    "t",t,"Ref",Ref,"Mis",Mis,"Learned",Learned)