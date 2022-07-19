using Plots, LaTeXStrings
using DifferentialEquations, LinearAlgebra

include("../LR_odes.jl")


## Definition of parameters
tau_s = 50
tau_us = 50*50

function asp_mod(t)
    if t < 2000
        2
    else
        2 +(t-2000)/760
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
p1=plot(sol.t/1000, sol[1,:],linewidth=1.5,legend=false)
ylabel!("V")

p1zoom=plot(sol.t/1000, sol[1,:],linewidth=1.5,legend=false,xlims=((ti3-50)/1000,(tf1+150)/1000))

# Input current
t=range(0.0,Tfinal,length=10000)
Ip2 = Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2)+I3*pulse.(t,ti3,tf3)
p2=plot(t/1000,Ip2,linewidth=3)
yticks!([-1,-0.95,-0.9,-0.85])
xlabel!(L"t [x $10^3$]")
ylabel!(L"i_{\rm{app}}")

p2zoom=plot(t/1000,Ip2,linewidth=3,xlims=((ti3-50)/1000,(tf1+150)/1000))
yticks!([-1,-0.95,-0.9,-0.85])
xlabel!(L"t [x $10^3$]")

l = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}] [c{1.0*w,0.7*h}
                d{1.0*w,0.3*h}]
]

# l2 = @layout [
#     a b; c d; e
# ]
CC = plot(p1,p2,p1zoom,p2zoom,layout=l,legend=false)

asp_mod_plot = t -> asp_mod(t*1000)
pgain = plot(t/1000,asp_mod_plot,color="darkred",legend=false)
xlabel!(L"t [x $10^3$]")
ylabel!(L"$\alpha_{\rm{s}}^+$")

# savefig(CC,"sec2_LR_spiking_mod.pdf")
# savefig(pgain,"sec2_LR_spiking_mod_pgain.pdf")

CC

l3 = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.2*h}
    c{1.0*w,0.1*h}] [d{1.0*w,0.7*h}
                e{1.0*w,0.2*h}
                f{1.0*w,0.1*h}]
]
pblank = plot(legend=false,grid=false,foreground_color_subplot=:white)  

#l2 = @layout [
#    [a{0.5*w,1.0*h} b{0.5*w,1.0*h}]; [c{0.5*w,0.6*h} d{0.5*w,0.6*h}]; e{1.0*w,0.2*h}
#]
#CC2 = plot(p1,p1zoom,p2,p2zoom,pgain,layout=l2,legend=false)
CC3 = plot(p1,p2,pgain,p1zoom,p2zoom,pblank,layout=l3,legend=false)
savefig(CC3,"sec2_spiking.pdf")