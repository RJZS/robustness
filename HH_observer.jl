using Plots: print
using DifferentialEquations, Random, Distributions, Plots, LinearAlgebra, DelimitedFiles
# Random.seed!(121)

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
γ = 2.0

# Modelling errors
# True values are: r_m = -40, r_h = -62, r_n = -53
err = 0.0# 5 # Maximum proportional error in r.
half_acts = (x_sample(-40, err),x_sample(-62, err),x_sample(-53, err))

Tfinal = 300.

# Noise-generated current
d = Normal(0,1.64)
n_per_t = 4
n = rand(d, Int(Tfinal*n_per_t)+2)
# Iapp = t -> -1 - 0*t 

# Interpolated noisy input for ODE solver
function noisy_input(Iconst, noise, n_per_t, t)
    dt = 1/n_per_t
    y0j = Int.(floor.(t.*n_per_t)).+1
    y0 = Iconst .+ noise[y0j]
    y1 = Iconst .+ noise[y0j.+1]
    t0 = t .- t.%dt
    t1 = t0 .+ dt

    y = y0 .+ (t .- t0).*(y1.-y0)./(t1-t0)
end
# nts = noisy_input(4,n,n_per_t,ts) # For LaTeXStrings

Iconst = 1.8
Iapp = t -> noisy_input(Iconst, n, n_per_t, t)

# Noise on measurements of v
sd = Normal(0, 20)
sn_per_t = 100
sn = rand(sd, Int(Tfinal*sn_per_t)+2)
sensor_noise = t -> noisy_input(0, sn, sn_per_t, t)

# Initial conditions
x₀ = [0 0 0 0]; 
x̂₀ = [-60 0.5 0.5 0.5];
θ̂₀ = [60 60 10];
P₀ = Matrix(I, 3, 3);
Ψ₀ = [0 0 0];

# Integration initial conditions and parameters
dt = 0.01
tspan = (0.,Tfinal)
z₀ = [x₀ x̂₀ θ̂₀ reshape(P₀,1,9) Ψ₀]
p = (Iapp,c,g,E,(α,γ),half_acts,sensor_noise)

# Integrate
prob = ODEProblem(HH_observer!,z₀,tspan,p)
sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8,saveat=0.1,maxiters=1e6)
t = sol.t
v = sol[1,1,:];
w = sol[1,2:4,:];
v̂ = sol[1,5,:];
ŵ = sol[1,6:8,:];
θ̂ = sol[1,9:11,:];
N = 15+9+3;

if save_data
    writedlm("./../data/HH_voltages.txt",  hcat(t,v,v̂), " ")
    writedlm("./../data/HH_parameters.txt",  hcat(t,θ̂'), " ")
end

## Plots
pltstart = 1
pltstartidx = 1+pltstart*10

plt0 = plot(t,v)
plt0 = plot!(t,v̂,linecolor="red",linestyle= :dash)

# gNa/c
plt1 = plot([pltstart,Tfinal],[g[1],g[1]],linecolor="black",linestyle=:dash,labels="gNa/c")
plt1 = plot!(t[pltstartidx:end],θ̂[1,pltstartidx:end],linecolor="red")

# gK/c
plt2 = plot([pltstart,Tfinal],[g[2],g[2]],linecolor="black",linestyle=:dash,labels="gK/c")
plt2 = plot!(t[pltstartidx:end],θ̂[2,pltstartidx:end],linecolor="red")

# gL/c
plt3 = plot([pltstart,Tfinal],[g[3],g[3]],linecolor="black",linestyle=:dash,labels="gL/c")
plt3 = plot!(t[pltstartidx:end],θ̂[3,pltstartidx:end],linecolor="red")

plt0