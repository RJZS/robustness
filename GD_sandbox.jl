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
gKCa=35.; # Calcium-activated potassium current maximal conductance
gCaL=0.; # L-type calcium current maximal conductance
gCaT=6.; # T-type calcium current maximal conductance
gH=0.; # H-current maximal conductance

# Initial conditions
x0 = init_neur(-70.);

Tfinal=6000.0
tspan=(0.0,Tfinal)

## Input current defition
# Constant current
Iapp=6.

# Current pulses
I1=-20. # Amplitude of first pulse
ti1=2000 # Starting time of first pulse
tf1=3000 # Ending time of first pulse
I2=-20. # Amplitude of second pulse
ti2=000. # Starting time of second pulse
tf2=0000. # Ending time of first pulse

## Current-clamp experiment
# Parameter vector for simulations
p=(Iapp,I1,I2,ti1,tf1,ti2,tf2,gNa,gKd,gAf,gAs,gKCa,gCaL,gCaT,gH,gl,gCaLinCa,gCaTinCa)

# Simulation
# Using the calcium observer
prob = ODEProblem(CBM_ODE,x0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
ylabel!("V")

p1zoom=plot(sol.t, sol[1,:],linewidth=1.5,legend=false,xlims=(tf1-100,tf1+2000))

# Input current
t=range(0.0,Tfinal,length=10000)
p2=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=3)
xlabel!("t")
ylabel!("I_ext")

p2zoom=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=3,xlims=(tf1-100,tf1+2000))
xlabel!("t")

l = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}] [c{1.0*w,0.7*h}
                d{1.0*w,0.3*h}]
]

CC = plot(p1,p2,p1zoom,p2zoom,layout=l,legend=false)