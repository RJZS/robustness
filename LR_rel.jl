# Reliability experiments on Luka's circuit

using Plots
using DifferentialEquations, LinearAlgebra

include("LR_odes.jl")


## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = -2
asp = 2
asn =  -1.5
ausp =  2

dfn = 0
dsp = 0
dsn = -0.88
dusp = 0

delta_h = -0.5
beta = 2

# delta_ests = (0, 0, -1.5, -1.5)
# delta_ests = (.4, -.3, -1.2, -1.9)
delta_ests = (0,0,-1.5,-1.5)


# Initial conditions
x0 = [-1.9 -1.9 -1.9]
u0 = x0

Tfinal= 16000.0
tspan=(0.0,Tfinal)
dt = 0.1
invdt = 10

# Noise-generated current
d = Normal(0,1)*6
noise = rand(d, round(Int, Tfinal/dt+1))
Iconst = -2.6

for i in eachindex(noise)
    i == 1 ? noise[i] = 0 : noise[i]=noise[i-1]+(noise[i]-noise[i-1])/1000
end
noise = noise .+ Iconst
plot(noise)

save("LR_rel_noise.jld","noise",noise)

# Simulation
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,noise,invdt)
prob = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,Euler(),adaptive=false,dt=dt)

p1=plot(sol.t, sol[1,:],linewidth=1.5,legend=false)
ylabel!("V")