# Reliability experiments on Luka's circuit

using Plots, LaTeXStrings
using DifferentialEquations, LinearAlgebra

include("../LR_odes.jl")

# Modulation fn
function asn_mod(t)
    if t < 5000
        -1.2
    elseif t > 20000
        -1.95
    else
        -1.2-(t-5000)/20000
    end
end

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = -2
asp = 2
asn =  t-> asn_mod(t)
ausp =  2

dfn = 0
dsp = 0
dsn = -1.5
dusp = -1.5

asn2 = -0.5

γ =2;
α = 0.0001;
Tfinal= 30000.0;
tspan=(0.0,Tfinal);

x0 = [-1.9 -1.9 -1.9];
xh0 = [-1 -1 -1];
θ̂₀ = -.5;
P₀ = 1;
Ψ₀ = 0;
x02 = [-1.2 -1.2 -1.2];
u0 = [x0 xh0 θ̂₀ P₀ Ψ₀ x02];

p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iapp,α,γ,asn2);
prob = ODEProblem(LR_observer_RefTrack!,u0,tspan,p) # Simulation without noise (ODE)
sol = solve(prob,Euler(),adaptive=false,dt=dt)

t = sol.t;
V = sol[1,:];
V2 = sol[10,:];

asn_est = sol[7,:];

p1 = plot(t,V,label="Reference",legend=:bottomleft)
plot!(t,V2,label="Controlled")
ylabel!("V")

p2 = plot(t,asn_mod,linewidth=2,label="True",legend=:bottomleft)
plot!(t, asn_est,linewidth=1.5,label="Estimate")
xlabel!("t")
ylabel!(L"\alpha_s^-")

CC = plot(p1, p2, layout = (2, 1))
savefig(CC, "sec3_LR_bursting.pdf")
CC
# # Truncated figures
# j = size(solObs)[3]
# i = round(Int,1*j/5)
# plot(sol.t[i:j], sol[7,i:j])
# plot!(sol.t[i:j], -sol[8,i:j]) # Plotting the negative so can compare!

# p1=plot(solRef.t, solRef[1,:])
