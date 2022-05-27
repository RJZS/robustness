using Plots, JLD
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
gNa=100.; # Sodium current maximal conductance
gKd=65.; # Delayed-rectifier potassium current maximal conductance
gAf=0.; # Fast A-type potassium current maximal conductance
gAs=0.; # Slow A-type potassium current maximal conductance
gKCa=8.; # Calcium-activated potassium current maximal conductance
gCaL=4.; # L-type calcium current maximal conductance
gCaT=0.5; # T-type calcium current maximal conductance
gH=0.; # H-current maximal conductance

# Observer parameters
α = 0.008
γ = 5

# Modelling errors
# True values are (45, 60, 85) for (mCaL, mCaT, hCaT)
# The gates are 
# (mNa, hNa, mKd, mAf, hAf, mAs, hAs, 
# mCaL, mCaT, hCaT, mH). The true values are
# (25, 40, 15, 80, 60, 60, 20, 45, 60, 85, 85, -30)
err = 0. # Maximum proportional error in observer model. Try eg 0.05 and 0.1.
# half_acts = (x_sample(45, err),x_sample(60, err),x_sample(85, err))
half_acts = (25*(1+err),40*(1+err),15*(1-err),
80*(1+err),60*(1-err),60*(1-err),
20*(1+err),
45*(1+err),60*(1-err),85*(1+err),85*(1+err),-30*(1-err))
# TODO: Try random error. Try error in other params.

# Errors in tau half act
# mNa 100 hNa 50 mKd 30
# mAf 100 hAf 100 mAs hAs 100 mCaL mCaT hCaT 30
# mH 30
err_t = 0.
half_act_taus = (100*(1+err_t),50*(1-err_t),30*(1-err_t),
            100*(1-err_t),100*(1+err_t),100*(1+err_t),100*(1+err_t),
            30*(1+err_t),30*(1+err_t),30*(1-err_t),30*(1+err_t))

# Initial conditions
x₀ = init_neur(-70.);
x̂₀ = [-60 0.4 0.4 0.4 0.4 0.4 0.5 0.3 0.5 0.6 0.1 0.5 0.2];
θ̂₀ = [.1 .1];
P₀ = Matrix(I, 2, 2);
Ψ₀ = [0 0 0 0]; # Flattened
u0 = [x₀ x̂₀ θ̂₀ reshape(P₀,1,4) Ψ₀]

Tfinal= 6000.0 # 14500.0
tspan=(0.0,Tfinal)

## Input current defition
# Constant current
#Iapp= 4. # Overwritten in the function by a hardcoded input.

# Noise-generated current
d = Normal(0,1)
n_per_t = 5
n = rand(d, Int(Tfinal*n_per_t)+2)
# Iapp = t -> -1 - 0*t 

# Interpolated noisy input for ODE solver
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
Iapp = t -> noisy_input(Iconst, n, n_per_t, t)
save("data.jld","noise",n,"n_per_t",n_per_t)
Iapp = t -> 4.

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
prob = ODEProblem(CBM_Ca_observer_with_v!,u0,tspan,p) # Simulation without noise (ODE)
# NOTE the above function is currently not using Ca, it's using Cah!!

# prob = ODEProblem(CBM_observer!,u0,tspan,p) # Simulation without noise (ODE)

sol = solve(prob,dtmax=0.1)
# sol = solve(prob,alg_hints=[:stiff],reltol=1e-8,abstol=1e-8)
# sol = solve(prob,AutoTsit5(Rosenbrock23()))
# using LSODA
# sol = solve(prob,lsoda(),reltol=1e-10,abstol=1e-10)

# Alternative from Thiago:
# sol = solve(prob,AutoTsit5(Rosenbrock23()),saveat=0.1)#,reltol=1e-8,abstol=1e-8

## Generation of figures 
# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
ylabel!("V")

p1b = plot(sol.t,sol[1,:])
plot!(sol.t, sol[14,:])

# Ca versus its estimate
# i = 220000
# j = lastindex(sol[1,:])
p2 = plot(sol.t, sol[13,:])
plot!(sol.t, sol[26,:])

# Parameter estimates
p3 = plot(sol.t,sol[27,:]) # gCaL
p4 = plot(sol.t,sol[28,:]) # gCaT

# Truncated figures
j = size(sol)[3]
i = round(Int,3*j/5)
p1t = plot(sol.t[i:j],sol[1,i:j],legend=false)
plot!(sol.t[i:j],sol[14,i:j])
ylabel!("V")

p2t = plot(sol.t[i:j], sol[13,i:j])
plot!(sol.t[i:j], sol[26,i:j])

# Parameter estimates
p3t = plot(sol.t[i:j],sol[27,i:j]) # gCaL
p4t = plot(sol.t[i:j],sol[28,i:j]) # gCaT

p1b