using Plots
using DifferentialEquations, LinearAlgebra

include("../LR_odes.jl")


## Definition of parameters
tau_s = 50
tau_us = 50*50

function asp_mod(t)
    if t < 2000
        2
    else
        2 +(t-2000)/3000
    end
end

afn = t -> -2
asp = t -> asp_mod(t)
asn =  t -> 0
ausp =  t -> 0

dfn = 0
dsp = 0
dsn = -0.88
dusp = 0

# Initial conditions
x0 = [-1 -1 -1]
u0 = x0

Tfinal= 3300.0
tspan=(0.0,Tfinal)

Iapp = -1
# Current pulses
I1=0.05 # Amplitude of first pulse
ti1=1000 # Starting time of first pulse
tf1=1020 # Ending time of first pulse
I2=0.15 # Amplitude of second pulse
ti2=2000 # Starting time of second pulse
tf2=3601 # Ending time of first pulse
I3 = 0.03 # Subthreshold input
ti3 = 900
tf3 = 920

# Parameter vector for simulations
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,
    Iapp,I1,I2,ti1,tf1,ti2,tf2,I3,ti3,tf3)

# Simulation
# Using the calcium observer
prob = ODEProblem(LR_ODE_mod!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
# # Voltage response
# p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
# plot!(sol.t,sol[2,:])
# plot!(sol.t,sol[3,:])
# ylabel!("V")

# Voltage response
p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
ylabel!("V")

p1zoom=plot(sol.t, sol[1,:],linewidth=1.5,legend=false,xlims=(ti3-50,tf1+150))

# Input current
t=range(0.0,Tfinal,length=10000)
p2=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2)+I3*pulse.(t,ti3,tf3),linewidth=3)
xlabel!("t")
ylabel!("I_ext")

p2zoom=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2)+I3*pulse.(t,ti3,tf3),linewidth=3,xlims=(ti3-50,tf1+150))
xlabel!("t")

l = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}] [c{1.0*w,0.7*h}
                d{1.0*w,0.3*h}]
]

CC = plot(p1,p2,p1zoom,p2zoom,layout=l,legend=false)

savefig(CC,"sec2_LR_spiking_mod.pdf")
