using Plots, LaTeXStrings
using DifferentialEquations, LinearAlgebra

include("../LR_odes.jl")

# Modulation fn
function asn2_mod(t)
    if t < 11000
        -1.6
    else
        -1.6 +(t-11000)/18000
    end
end

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = t -> -2
asp = t -> 2
asn =  t -> -1.6
ausp =  t -> 2

dfn = 0
dsp = 0
dsn = -0.88
dusp = 0

afn2 = t -> -2
asp2 = t -> 2
asn2 =  t -> asn2_mod(t)
ausp2 =  t -> 2

asyn21 = t -> -0.2
asyn12 = t -> -0.2

deltasyn = -1
delta_h = -0.5

# Initial conditions
x0 = [-0.2 -0.2 -0.2]
x02 = [0 0 0]
u0 = [x0 x02]

Tfinal= 26000.0
tspan=(0.0,Tfinal)

Iapp = -2.2 # -0.8
# Current pulses
I1=0 # Amplitude of first pulse
ti1=4500 # Starting time of first pulse
tf1=6000 # Ending time of first pulse
I2=0 # Amplitude of second pulse
ti2=9000 # Starting time of second pulse
tf2=16001 # Ending time of first pulse

Iapp2 = -2.5
I3=0.1 # Amplitude of first pulse
ti3=6100 # Starting time of first pulse
tf3=6401 # Ending time of first pulse
I4=0.2 # Amplitude of second pulse
ti4=11000 # Starting time of second pulse
tf4=26001 # Ending time of first pulse
I5 = 0.08 # For subthreshold response
ti5 = 4800
tf5 = 5000

# Parameter vector for simulations
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,
    Iapp,I1,I2,ti1,tf1,ti2,tf2,afn2,asp2,asn2,ausp2,Iapp2,
    asyn21,asyn12,deltasyn,I3,ti3,tf3,delta_h,I4,ti4,tf4,I5,ti5,tf5)

# Simulation
# Using the calcium observer
prob = ODEProblem(LR_II_ODE_with_inact_mod!,u0,tspan,p) # Simulation without noise (ODE)
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
xticks!([0, 10000, 20000])
ylabel!("V")

p1zoom=plot(sol.t, sol[1,:],linewidth=1.5,legend=false,xlims=(ti3-800,tf3+2620))
plot!(sol.t, sol[4,:],linewidth=1.5,legend=false,xlims=(ti5-800,tf3+2620))

# Input current
t=range(0.0,Tfinal,length=10000)
p2=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=1.5)
plot!(t,Iapp2 .+I3*pulse.(t,ti3,tf3)+I4*pulse.(t,ti4,tf4)+I5*pulse.(t,ti5,tf5),linewidth=1.5)
xticks!([0, 10000, 20000])
xlabel!("t")
ylabel!(L"i_{\rm{app}}")

p2zoom=plot(t,Iapp .+I1*pulse.(t,ti1,tf1)+I2*pulse.(t,ti2,tf2),linewidth=1.5,xlims=(ti5-800,tf3+2620))
plot!(t,Iapp2 .+I3*pulse.(t,ti3,tf3)+I4*pulse.(t,ti4,tf4)+I5*pulse.(t,ti5,tf5),linewidth=1.5,xlims=(ti5-800,tf3+2620))
xlabel!("t")

l = @layout [
    [a{1.0*w,0.7*h}
    b{1.0*w,0.3*h}] [c{1.0*w,0.7*h}
                d{1.0*w,0.3*h}]
]

CC = plot(p1,p2,p1zoom,p2zoom,layout=l,legend=false)

savefig(CC,"sec2_LR_II_mod.pdf")

CC