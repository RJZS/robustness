using Plots
using DifferentialEquations, LinearAlgebra
using Random, Distributions
# Random.seed!(123)

include("GD_odes.jl")

## Constant simulation parameters

## Definition of reversal potential values. 
const VNa = 40.; # Sodium reversal potential
const VK = -90.; # Potassium reversal potential
const VCa = 120.; # Calcium reversal potential
const VH= -40.; # Reversal potential for the H-current (permeable to both sodium and potassium ions)
const Vl = -50.; # Reversal potential of leak channels

const C=0.1; # Membrane capacitance
αCa=0.1; # Calcium dynamics (L-current)
β=0.04 # Calcium dynamics (T-current)

gl=0.3; # Leak current maximal conductance

gNa=96.325; # Sodium current maximal conductance
gKd=62.517; # Delayed-rectifier potassium current maximal conductance
gAf=0.; # Fast A-type potassium current maximal conductance
gAs=0.; # Slow A-type potassium current maximal conductance
gKCa=8.1848; # Calcium-activated potassium current maximal conductance
gCaL=3.8154; # L-type calcium current maximal conductance
gCaT=0.54045; # T-type calcium current maximal conductance
gH=0.; # H-current maximal conductance

# Initial conditions
u0 = init_neur(-60.);

Tfinal= 20000.0 # 14500.0
tspan=(0.0,Tfinal)

## Input current defition
# Constant current
#Iapp= 4. # Overwritten in the function by a hardcoded input.

# Noise-generated current
function noisy_input(Iconst, noise, n_per_t, t)
    dt = 1/n_per_t
    y0j = Int.(floor.(t.*n_per_t)).+1
    y0 = Iconst .+ noise[y0j]
    y1 = Iconst .+ noise[y0j.+1]
    t0 = t .- t.%dt
    t1 = t0 .+ dt

    y = y0 .+ (t .- t0).*(y1.-y0)./(t1-t0)
end
# nts = noisy_input(4,n,n_per_t,ts) # For LaTeXStrings

Iconst = -1.5
n = load("data.jld")["noise"]
n_per_t = load("data.jld")["n_per_t"]
Iapp = t -> noisy_input(Iconst, n, n_per_t, t)
# Iapp = t -> 4.

# Current pulses
I1=0. # Amplitude of first pulse
ti1=300 # Starting time of first pulse
tf1=302 # Ending time of first pulse
I2=0. # Amplitude of second pulse
ti2=350 # Starting time of second pulse
tf2=370 # Ending time of first pulse

# Have hardcoded the input current into the ODE fn.
# # High frequency noise input 
# Isines = .5*sin.(0.01*sol.t)+0.5*sin.(0.05*sol.t)

## Current-clamp experiment
# Parameter vector for simulations
p=(Iapp,I1,I2,ti1,tf1,ti2,tf2,
gNa,gKd,gAf,gAs,gKCa,gCaL,gCaT,gH,gl,half_acts,half_act_taus)

# Simulation
# Using the calcium observer
prob = ODEProblem(CBM_ODE,u0,tspan,p) # Simulation without noise (ODE)
# Note the above function is currently not using Ca, it's using Cah!!

# prob = ODEProblem(CBM_observer!,u0,tspan,p) # Simulation without noise (ODE)

sol = solve(prob,dtmax=0.1,saveat=0.1)
# sol = solve(prob,alg_hints=[:stiff],reltol=1e-8,abstol=1e-8)
# sol = solve(prob,AutoTsit5(Rosenbrock23()))
# using LSODA
# sol = solve(prob,lsoda(),reltol=1e-10,abstol=1e-10)

# Alternative from Thiago:
# sol = solve(prob,AutoTsit5(Rosenbrock23()),saveat=0.1)#,reltol=1e-8,abstol=1e-8

## Generation of figures 
# Voltage response
t = sol.t
V = sol[1,:]
Ca_ = sol[13,:] # To distinguish from Ca in the other script.
p1=plot(sol.t, V,linewidth=1.5,legend=false)
ylabel!("V")


# Ca versus its estimate
# i = 220000
# j = lastindex(sol[1,:])
p2 = plot(sol.t, Ca_[13,:])


# Truncated figures
j = size(sol)[3]
i = round(Int,3*j/5)
p1t = plot(sol.t[i:j],V[i:j],legend=false)
ylabel!("V")

p2t = plot(sol.t[i:j], Ca_[i:j])
p1

p_compare = plot(t,V)
plot!(t,Vref)