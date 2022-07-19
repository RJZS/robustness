using Plots, LaTeXStrings
using DifferentialEquations, LinearAlgebra

include("../LR_odes.jl")

# Modulation fn
function asn2_mod(t)
    if t < 11000
        -1.5
    else
        -1.5 +(t-11000)/18000
    end
end

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = t -> -2
asp = t -> 2
asn =  t -> -1.5
ausp =  t -> 1.5

dfn = 0
dsp = 0
dsn = -0.9
dusp = -2.8

afn2 = t -> -2
asp2 = t -> 2
asn2 =  t -> asn2_mod(t)
ausp2 =  t -> 1.5

asyn21 = t -> -2 # -0.2
asyn12 = t -> -2 # -0.2

deltasyn = -1
delta_h = -0.5

# Initial conditions
x0 = [-2 -2 -2]
x02 = [-2 -2 -2]
u0 = [x0 x02]

Tfinal= 20000.0
tspan=(0.0,Tfinal)

Iapp = 0.2 # -0.8
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
I4=0#.2 # Amplitude of second pulse
ti4=11000 # Starting time of second pulse
tf4=26001 # Ending time of first pulse
I5 = 0#.08 # For subthreshold response
ti5 = 4800
tf5 = 5000

# Parameter vector for simulations
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,
    Iapp,I1,I2,ti1,tf1,ti2,tf2,afn2,asp2,asn2,ausp2,Iapp2,
    asyn21,asyn12,deltasyn,I3,ti3,tf3,delta_h,I4,ti4,tf4,I5,ti5,tf5)

# Simulation
# Using the calcium observer
prob = ODEProblem(LR_II_ODE_mod!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,dtmax=0.1)


## Generation of figures 
# # Voltage response
# p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
# plot!(sol.t,sol[2,:])
# plot!(sol.t,sol[3,:])
# ylabel!("V")

# Voltage response
p1=plot(sol.t/1000, sol[1,:],linewidth=1.5,legend=false)
plot!(sol.t/1000, sol[4,:])
xticks!([0, 10, 20])
ylabel!("V")

xlims = ((ti3-800)/1000,(tf3+2500)/1000)
p1zoom=plot(sol.t/1000, sol[1,:],linewidth=1.5,legend=false,xlims=xlims)
plot!(sol.t/1000, sol[4,:],linewidth=1.5,legend=false,xlims=xlims)
xticks!([6, 8, 10, 12])

# Input current
t=range(0.0,Tfinal,length=10000)
Ip2_N1=Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2)
Ip2_N2=Iapp2 .+I3*pulse.(t,ti3,tf3)+I4*pulse.(t,ti4,tf4)+I5*pulse.(t,ti5,tf5)
p2=plot(t/1000,Ip2_N1,linewidth=1.5)
plot!(t/1000,Ip2_N2,linewidth=1.5)
xticks!([0, 10, 20])
xlabel!(L"t [x $10^3$]")
ylabel!(L"i_{\rm{app}}")

p2zoom=plot(t/1000,Ip2_N1,linewidth=1.5,xlims=xlims)
plot!(t/1000,Ip2_N2,linewidth=1.5,xlims=xlims)
xticks!([6, 8, 10, 12])
xlabel!(L"t [x $10^3$]")

l = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}] [c{1.0*w,0.7*h}
                d{1.0*w,0.3*h}]
]

CC = plot(p1,p2,p1zoom,p2zoom,layout=l,legend=false)

asn2_mod_plot = t -> asn2_mod(t*1000)
pgain = plot(t/1000,asn2_mod_plot,color="darkred",legend=false)
xlabel!(L"t [x $10^3$]")
yticks!([-1,-1.25,-1.5])
ylabel!(L"$\alpha_{\rm{s},2}^-$")

savefig(CC,"sec2_LR_II_mod.pdf")
savefig(pgain,"sec2_LR_II_mod_pgain.pdf")

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
savefig(CC2,"sec2_II.pdf")