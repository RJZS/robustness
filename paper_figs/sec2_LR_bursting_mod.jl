using Plots, LaTeXStrings
using DifferentialEquations, LinearAlgebra

include("../LR_odes.jl")

# Modulation fn
function asn_mod(t)
    if t < 14000
        -1.5
    else
        -1.8
    end
end

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = t -> -2
asp = t -> 2
asn = t -> asn_mod(t)
ausp = t -> 2

dfn = 0
dsp = 0
dsn = -0.88
dusp = 0

# Initial conditions
x0 = [-1.9 -1.9 -1.9]
u0 = x0

Tfinal= 18000.0
tspan=(0.0,Tfinal)

Iapp = -2.6
# Current pulses
I1=0.2 # Amplitude of first pulse
ti1=3000 # Starting time of first pulse
tf1=3100 # Ending time of first pulse
I2=0.2 # Amplitude of second pulse
ti2=8000 # Starting time of second pulse
tf2=26001 # Ending time of first pulse
I3 = 0.05 # Subthreshold input
ti3 = 2600
tf3 = 2700

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

p1zoom=plot(sol.t, sol[1,:],linewidth=1.5,legend=false,xlims=(ti3-50,tf1+1250))
xticks!([3000, 3500, 4000])

# Input current
t=range(0.0,Tfinal,length=10000)
p2=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2)+I3*pulse.(t,ti3,tf3),linewidth=3)
yticks!([-2.6, -2.5, -2.4])
xlabel!("t")
ylabel!(L"i_{\rm{app}}")

p2zoom=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2)+I3*pulse.(t,ti3,tf3),linewidth=3,xlims=(ti3-50,tf1+1250))
yticks!([-2.6, -2.5, -2.4])
xticks!([3000, 3500, 4000])
xlabel!("t")

l = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}] [c{1.0*w,0.7*h}
                d{1.0*w,0.3*h}]
]

CC = plot(p1,p2,p1zoom,p2zoom,layout=l,legend=false)

savefig(CC,"sec2_LR_bursting_mod.pdf")

CC