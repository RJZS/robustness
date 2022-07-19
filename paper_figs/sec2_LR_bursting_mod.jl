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
p1=plot(sol.t/1000, sol[1,:],linewidth=1.5,legend=false)
ylabel!("V")

p1zoom=plot(sol.t/1000, sol[1,:],linewidth=1.5,legend=false,xlims=((ti3-50)/1000,(tf1+1250)/1000))
xticks!([3, 3.5, 4])

# Input current
t=range(0.0,Tfinal,length=10000)
Ip2 = Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2)+I3*pulse.(t,ti3,tf3)
p2=plot(t/1000,Ip2,linewidth=3)
yticks!([-2.6, -2.5, -2.4])
xlabel!(L"t [x $10^3$]")
ylabel!(L"i_{\rm{app}}")

p2zoom=plot(t/1000,Ip2,linewidth=3,xlims=((ti3-50)/1000,(tf1+1250)/1000))
yticks!([-2.6, -2.5, -2.4])
xticks!([3, 3.5, 4])
xlabel!(L"t [x $10^3$]")

l = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}] [c{1.0*w,0.7*h}
                d{1.0*w,0.3*h}]
]

CC = plot(p1,p2,p1zoom,p2zoom,layout=l,legend=false)

asn_mod_plot = t -> asn_mod(t*1000)
pgain = plot(t/1000,asn_mod_plot,color="darkred",legend=false)
yticks!([-1.8, -1.6])
xlabel!(L"t [x $10^3$]")
ylabel!(L"$\alpha_{\rm{s}}^-$")

savefig(CC,"sec2_LR_bursting_mod.pdf")
savefig(pgain,"sec2_LR_bursting_mod_pgain.pdf")

CC

l2 = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.2*h}
    c{1.0*w,0.1*h}] [d{1.0*w,0.7*h}
                e{1.0*w,0.2*h}
                f{1.0*w,0.1*h}]
]
pblank = plot(legend=false,grid=false,foreground_color_subplot=:white)  
CC2 = plot(p1,p2,pgain,p1zoom,p2zoom,pblank,layout=l3,legend=false)
savefig(CC2,"sec2_bursting.pdf")