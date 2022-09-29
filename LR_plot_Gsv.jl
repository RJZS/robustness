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


# Initial conditions
x0 = [-1.9 -1.8 -1.8]
Ψ₀ = [0 0 0 0];
bf1 = 0; bf2 = 0; Gdv = 0;
u0 = [x0 Ψ₀ Ψ₀ bf1 bf2 Gdv]

Tfinal= 14000.0
tspan=(0.0,Tfinal)

γ = 10
β = 2

Iapp = t -> -2.

# Parameter vector for simulations
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iapp,γ,β)

# Simulation
# Using the calcium observer
prob = ODEProblem(LR_plot_Gsv!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
p1 = plot(sol.t, sol[1,:])
p2 = plot(sol.t, sol[14,:]) # Gsv
p3 = plot(sol.t, sol[1,:])
plot!(sol.t, sol[14,:])

p2