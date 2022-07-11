using Plots
using DifferentialEquations, LinearAlgebra

include("../LR_odes.jl")


## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = -2
asp = 2
asn =  -1.5
ausp =  1.5

dfn = 0
dsp = 0
dsn = -0.9
dusp = -2.8 # -0.88

afn2 = -2
asp2 = 2
asn2 =  -1.5
ausp2 =  1.5

asyn21 = -2 # -0.4
asyn12 = -2 # -0.1

deltasyn = -1

# Initial conditions
x0 = [-2 -2 -2]
x02 = [-2 -2 -2]
u0 = [x0 x02]

Tfinal= 18000.0
tspan=(0.0,Tfinal)

Iapp = 0.2
# Current pulses
I1=0.2 # Amplitude of first pulse
ti1=1800 # Starting time of first pulse
tf1=2500 # Ending time of first pulse
I2=-0.5 # Amplitude of second pulse
ti2=6000 # Starting time of second pulse
tf2=8001 # Ending time of first pulse

Iapp2 = 0.2
I3=0.#6 # Amplitude of first pulse
ti3=6000 # Starting time of first pulse
tf3=10001 # Ending time of first pulse

# Parameter vector for simulations
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,
    Iapp,I1,I2,ti1,tf1,ti2,tf2,afn2,asp2,asn2,ausp2,Iapp2,
    asyn21,asyn12,deltasyn,I3,ti3,tf3)

# Simulation
# Using the calcium observer
prob = ODEProblem(LR_II_ODE!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
# # Voltage response
# p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
# plot!(sol.t,sol[2,:])
# plot!(sol.t,sol[3,:])
# ylabel!("V")

# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
plot!(sol.t, sol[4,:])
ylabel!("V")

p1zoom=plot(sol.t, sol[1,:],linewidth=1.5,legend=false,xlims=(ti1-50,tf1+320))

# Input current
t=range(0.0,Tfinal,length=10000)
p2=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=1.5)
plot!(t,Iapp2 .+I3*pulse.(t,ti3,tf3),linewidth=1.5)
xlabel!("t")
ylabel!("I_ext")

p2zoom=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=3,xlims=(ti1-50,tf1+320))
xlabel!("t")

l = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}] [c{1.0*w,0.7*h}
                d{1.0*w,0.3*h}]
]

CC = plot(p1,p2,p1zoom,p2zoom,layout=l,legend=false)

# savefig(CC,"sec2_LR_II.pdf")