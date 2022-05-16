using Plots
using DifferentialEquations, LinearAlgebra

include("LR_odes.jl")


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
u0 = [-1.9 -1.8 -1.8]

Tfinal= 10000.0
tspan=(0.0,Tfinal)

Iapp = -2.

# Parameter vector for simulations
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iapp)

# Simulation
# Using the calcium observer
prob = ODEProblem(LR_ODE!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
plot!(sol.t,sol[2,:])
plot!(sol.t,sol[3,:])
ylabel!("V")
