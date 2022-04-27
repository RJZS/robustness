using Plots
using DifferentialEquations

include("GD_odes.jl")

## Constant simulation parameters

## Definition of reversal potential values. 
const VNa = 40.; # Sodium reversal potential
const VK = -90.; # Potassium reversal potential
const VCa = 120.; # Calcium reversal potential
const VH= -40.; # Reversal potential for the H-current (permeable to both sodium and potassium ions)
const Vl = -50.; # Reversal potential of leak channels

const C=0.1; # Membrane capacitance
const αCa=0.1; # Calcium dynamics (L-current)
const β=0.1 # Calcium dynamics (T-current)

gl=0.3; # Leak current maximal conductance
gNa=100.; # Sodium current maximal conductance
gKd=65.; # Delayed-rectifier potassium current maximal conductance
gAf=0.; # Fast A-type potassium current maximal conductance
gAs=0.; # Slow A-type potassium current maximal conductance
gKCa=8.; # Calcium-activated potassium current maximal conductance
gCaL=4.; # L-type calcium current maximal conductance
gCaT=0.5; # T-type calcium current maximal conductance
gH=0.; # H-current maximal conductance

gCaLinCa = 4.; # Hardcoded value of gCaL for use in [Ca] dynamics
gCaTinCa = 0.5;

# Observer parameters
α = 0.001
γ = 10.0

# Initial conditions
x₀ = init_neur(-70.);
x̂₀ = [-60 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5];
θ̂₀ = [1 1];
P₀ = Matrix(I, 2, 2);
Ψ₀ = [0 0];
u0 = [x₀ x̂₀ θ̂₀ reshape(P₀,1,4) Ψ₀]

Tfinal=20000.0
tspan=(0.0,Tfinal)

## Input current defition
# Constant current
Iapp=4.

# Current pulses
I1=0. # Amplitude of first pulse
ti1=300 # Starting time of first pulse
tf1=302 # Ending time of first pulse
I2=0. # Amplitude of second pulse
ti2=350 # Starting time of second pulse
tf2=370 # Ending time of first pulse

## Current-clamp experiment
# Parameter vector for simulations
p=(Iapp,I1,I2,ti1,tf1,ti2,tf2,gNa,gKd,gAf,gAs,gKCa,gCaL,gCaT,gH,gl,gCaLinCa,gCaTinCa)

# Simulation
# Using the calcium observer
prob = ODEProblem(CBM_Ca_observer!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
ylabel!("V")

# Ca versus its estimate
p2 = plot(sol.t, sol[13,:])
plot!(sol.t, sol[26,:])