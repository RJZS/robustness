using Plots
using DifferentialEquations, LinearAlgebra

include("LR_odes.jl")

# Trying out the bandpass filter.

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = -2
asp = 2
asn =  -1.5
ausp =  1.5

dfn = 0
dsp = 0
dsn = -1.5
dusp = -1.5

# delta_ests = (0, 0, -1.5, -1.5)
# delta_ests = (.4, -.3, -1.2, -1.9)
delta_ests = (0,0,-1.5,-1.5)

# Initial conditions
x0 = [-1.9 -1.8 -1.8]
xh0 = [-1 -1 -1]
θ̂₀ = [.1 .1 .1 .1];
P₀ = Matrix(I, 4, 4);
Ψ₀ = [0 0 0 0];
u0 = [x0 xh0 θ̂₀ P₀[:]' Ψ₀ Ψ₀]

Tfinal= 19000.0
tspan=(0.0,Tfinal)

γ = 0.2
α = 0.0004
β = 1
g = 5

Iapp = t -> -2.

# Parameter vector for simulations
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iapp,delta_ests,α,γ,β,g,[tau_s,tau_us])

# Simulation
# Using the calcium observer
prob = ODEProblem(LR_observer_newfilter!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
plot!(sol.t,sol[4,:])
ylabel!("V")

pe = plot(sol.t, sol[1,:].-sol[4,:])

p2 = plot(sol.t,sol[7,:])
p3 = plot(sol.t,sol[8,:])
p4 = plot(sol.t,sol[9,:])
p5 = plot(sol.t,sol[10,:])

pe