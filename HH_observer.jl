using Plots: print
using DifferentialEquations, Random, Distributions, Plots, LinearAlgebra, DelimitedFiles

# Flag for saving data to .txt files 
save_data = false

include("HH_odes.jl")

# True Parameters 
c = 1.
g = (120.,36.,0.3)
E = (55.,-77.,-54.4)
Iapp = t -> 2 + sin(2*pi/10*t)

# Observer parameters
α = 0.5
γ = 5.0

# Modelling errors
# True values are: r_m = -40, r_h = -62, r_n = -53
err = 0.04 # Maximum proportional error in r.
half_acts = (x_sample(-40, err),x_sample(-62, err),x_sample(-53, err))

# Initial conditions
x₀ = [0 0 0 0]; 
x̂₀ = [-60 0.5 0.5 0.5];
θ̂₀ = [60 60 10 0 0 0 0];
P₀ = Matrix(I, 7, 7);
Ψ₀ = [0 0 0 0 0 0 0];

# Integration initial conditions and parameters
dt = 0.01
Tfinal = 200. # 100.
tspan = (0.,Tfinal)
z₀ = [x₀ x̂₀ θ̂₀ reshape(P₀,1,49) Ψ₀]
p = (Iapp,c,g,E,(α,γ),half_acts)

# Integrate
prob = ODEProblem(HH_observer!,z₀,tspan,p)
sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8,saveat=0.1,maxiters=1e6)
t = sol.t
v = sol[1,1,:];
w = sol[1,2:4,:];
v̂ = sol[1,5,:];
ŵ = sol[1,6:8,:];
θ̂ = sol[1,9:15,:];
N = 15+49+7;

if save_data
    writedlm("./../data/HH_voltages.txt",  hcat(t,v,v̂), " ")
    writedlm("./../data/HH_parameters.txt",  hcat(t,θ̂'), " ")
end

## Plots

plt0 = plot(t,v)
plt0 = plot!(t,v̂,linecolor="red",linestyle= :dash)

# gNa/c
plt1 = plot([0,Tfinal],[g[1]/c,g[1]/c],linecolor="black",linestyle=:dash,labels="gNa/c")
plt1 = plot!(t,θ̂[1,:],linecolor="red")

# gK/c
plt2 = plot([0,Tfinal],[g[2]/c,g[2]/c],linecolor="black",linestyle=:dash,labels="gK/c")
plt2 = plot!(t,θ̂[2,:],linecolor="red")

# gL/c
plt3 = plot([0,Tfinal],[g[3]/c,g[3]/c],linecolor="black",linestyle=:dash,labels="gL/c")
plt3 = plot!(t,θ̂[3,:],linecolor="red")

# gNa*ENa/c
plt4 = plot([0,Tfinal],[g[1]*E[1]/c,g[1]*E[1]/c],linecolor="black",linestyle=:dash,labels="gNa*ENa/c")
plt4 = plot!(t,θ̂[4,:],linecolor="red")

# gK*EK/c
plt5 = plot([0,Tfinal],[g[2]*E[2]/c,g[2]*E[2]/c],linecolor="black",linestyle=:dash,labels="gK*EK/c")
plt5 = plot!(t,θ̂[5,:],linecolor="red")

# gL*EL/c
plt6 = plot([0,Tfinal],[g[3]*E[3]/c,g[3]*E[3]/c],linecolor="black",linestyle=:dash,labels="gL*EL/c")
plt6 = plot!(t,θ̂[6,:],linecolor="red")

# 1/c
plt7 = plot([0,Tfinal],[1/c,1/c],linecolor="black",linestyle=:dash,labels="1/c")
plt7 = plot!(t,θ̂[7,:],linecolor="red")