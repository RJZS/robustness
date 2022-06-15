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

delta_h = -0.5
beta = 2

# delta_ests = (0, 0, -1.5, -1.5)
# delta_ests = (.4, -.3, -1.2, -1.9)
delta_ests = (0,0,-1.5,-1.5)

# Initial conditions
x0 = [-1.9 -1.8 -1.8]
xh0 = [-1 -1 -1]
θ̂₀ = .1;
P₀ = 1;
Ψ₀ = 0;
u0 = [x0 xh0 θ̂₀ P₀ Ψ₀]

Tfinal= 16000.0
tspan=(0.0,Tfinal)

γ = 2
α = 0.01 # 0.0004

Iapp = -2.

# Parameter vector for simulations
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iapp,delta_ests,delta_h,beta)

# Simulation
# Using the calcium observer
prob = ODEProblem(LR_observer!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
plot!(sol.t,sol[2,:])
plot!(sol.t,sol[3,:])
ylabel!("V")

p2 = plot(sol.t,sol[7,:])