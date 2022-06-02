using Plots: print
using DifferentialEquations, Random, Distributions, Plots, LinearAlgebra, DelimitedFiles
# Random.seed!(121)

# Flag for saving data to .txt files 
save_data = false

include("HH_odes.jl")

# True Parameters 
c = 1.
g = (120.,36.,0.3)
errg = 0.001
g2 = (x_sample(120,errg),x_sample(36,errg),x_sample(0.3,errg))
E = (55.,-77.,-54.4)
Iapp = t -> 2 + sin(2*pi/10*t)

# Observer parameters
α = 0.5
γ = 2.0

# Modelling errors
# True values are: r_m = -40, r_h = -62, r_n = -53
err = 0.002 # Maximum proportional error in r.
half_acts = (-40, -62, -53)
half_acts2 = (x_sample(-40, err),x_sample(-62, err),x_sample(-53, err))

Tfinal = 200.

# Noise-generated current
d = Normal(0,1.62)
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

Iconst = 2
Iapp = t -> noisy_input(Iconst, n, n_per_t, t)
# Iapp = t -> 2 + sin(2*pi/10*t)
Iapp = t -> Iconst

# Initial conditions
x₀ = [0 0 0 0];

# Integration initial conditions and parameters
dt = 0.01
tspan = (0.,Tfinal)
z₀ = x₀
p = (Iapp,c,g,E,half_acts)
p2 = (Iapp,c,g2,E,half_acts2)

# Integrate
prob = SDEProblem(HH_ode!,HH_ode_noise!,z₀,tspan,p)
sol = solve(prob,EM(),dt=dt)

t = sol.t
v = sol[1,1,:];
# v2 = sol2[1,1,:];
w = sol[1,2:4,:];
N = 15+9+3;

if save_data
    writedlm("./../data/HH_voltages.txt",  hcat(t,v,v̂), " ")
    writedlm("./../data/HH_parameters.txt",  hcat(t,θ̂'), " ")
end

## Plots
pltstart = 1
pltstartidx = 1+pltstart*10

plt0 = plot(t,v)

plt0

# pltrel = plot(t,v)
# plot!(t,v2)