# Simulations on Drion's model for section 2 of the paper.

using Plots, JLD
using DifferentialEquations, LinearAlgebra

include("../GD_odes.jl")

## Constant simulation parameters

## Definition of reversal potential values. 
const VNa = 40.; # Sodium reversal potential
const VK = -90.; # Potassium reversal potential
const VCa = 120.; # Calcium reversal potential
const VH= -40.; # Reversal potential for the H-current (permeable to both sodium and potassium ions)
const Vl = -50.; # Reversal potential of leak channels
Esyn = -90

const C=0.1; # Membrane capacitance
αCa=0.1; # Calcium dynamics (L-current)
β=0.1 # Calcium dynamics (T-current)

gl=0.3; # Leak current maximal conductance
gNa=100.; # Sodium current maximal conductance
gKd=65.; # Delayed-rectifier potassium current maximal conductance
gAf=0.; # Fast A-type potassium current maximal conductance
gAs=0.; # Slow A-type potassium current maximal conductance
gKCa=35.; # Calcium-activated potassium current maximal conductance
gCaL=2.5; # L-type calcium current maximal conductance
gCaT=6.2; # T-type calcium current maximal conductance
gH=0.; # H-current maximal conductance

# N2
gl2=0.3; # Leak current maximal conductance
gNa2=100.; # Sodium current maximal conductance
gKd2=60.; # Delayed-rectifier potassium current maximal conductance
gAf2=0.; # Fast A-type potassium current maximal conductance
gAs2=0.; # Slow A-type potassium current maximal conductance
gKCa2=35.; # Calcium-activated potassium current maximal conductance
gCaL2=2.; # L-type calcium current maximal conductance
gCaT2=6.; # T-type calcium current maximal conductance
gH2=0.07; # H-current maximal conductance

gsyn21 = 1.
gsyn12 = 4.

# Initial conditions
x₀ = init_neur(-70.)
x₀2 = init_neur(-65.)
u0 = [x₀ x₀2 0 0] # The two synapses are at the end.

Tfinal= 6800.0 # 14500.0
tspan=(0.0,Tfinal)

Iapp = 4.
# Current pulses
I1=-4. # Amplitude of first pulse
ti1=2000 # Starting time of first pulse
tf1=2500 # Ending time of first pulse
I2=0. # Amplitude of second pulse
ti2=5000 # Starting time of second pulse
tf2=7501 # Ending time of first pulse

Iapp2 = 4. # N2 receives a constant current.

## Current-clamp experiment
# Parameter vector for simulations
p=(Iapp,I1,I2,ti1,tf1,ti2,tf2,
gNa,gKd,gAf,gAs,gKCa,gCaL,gCaT,gH,gl,
gNa2,gKd2,gAf2,gAs2,gKCa2,gCaL2,gCaT2,gH2,gl2,
gsyn21,gsyn12,Iapp2)

# Simulation
prob = ODEProblem(CBM_II_sec2,u0,tspan,p)
sol = solve(prob,dtmax=0.1,saveat=0.1)

# Extract output variables
t = sol.t; V = sol[1,:];

## Generation of figures 
# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
ylabel!("V")

p1zoom=plot(sol.t, sol[1,:],linewidth=1.5,legend=false,xlims=(ti1-50,tf1+320))

# Input current
t=range(0.0,Tfinal,length=10000)
p2=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=3)
xlabel!("t")
ylabel!("I_ext")

p2zoom=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=3,xlims=(ti1-50,tf1+320))
xlabel!("t")

p3=plot(sol.t, sol[14,:],linewidth=1.5,legend=false)
ylabel!("V")

p3zoom=plot(sol.t, sol[14,:],linewidth=1.5,legend=false,xlims=(ti1-50,tf1+320))

p5 = plot(sol.t, sol[1,:])
plot!(sol.t, sol[14,:])

p6=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=1.5)
hline!([Iapp2],linewidth=1.5,linestyle=:dash)
xlabel!("t")
ylabel!("I_ext")

l = @layout [
    a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}
]

CC = plot(p5,p6,layout=l,legend=false)
#xlims!(0,Tfinal)
#ylims!(-80,-60)

savefig(CC,"sec2_GD_II.pdf")