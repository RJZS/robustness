using Plots: print
using DifferentialEquations, Random, Distributions, Plots, LinearAlgebra, DelimitedFiles

save_data = false
include("HH_odes.jl")
dt = 0.01
Tfinal = 100.
tspan = (0.,Tfinal)

# Define bump function for input
α = 2.
τ = 5.
γ(t) = pdf.(Gamma(α,τ),t)

# Define input
I0 = 0.
a₁ = 15.; t₁ = 10.;
a₂ = 25.; t₂ = t₁ + 40.;
Iapp(t) = I0 + a₁.*γ(t.-t₁) + a₂.*γ(t.-t₂)

# Initial conditions
v0 = -64.
(taumNa,sigmNa) = gating_m(v0);
(tauhNa,sighNa) = gating_h(v0);
(taumK,sigmK) = gating_n(v0);
x0 = [-64. sigmNa sighNa sigmK]

# True Parameters 
c = 1.
g = (120.,36.,0.3)
E = (55.,-77.,-54.4)

# "True system" parameters
p = (Iapp,c,g,E)

# Obtain solution of the true model 
prob = ODEProblem(HH_ode!,x0,tspan,p)
sol = solve(prob,Tsit5(),saveat=dt,maxiters=1e6)
t = sol.t
v = sol[1,1,:];

# Obtain solutions of the estimator model for different parameters ĝNa
# and compute associated cost function V
dg = 0.1
ĝNa = 100.:dg:130.
V = zeros(length(ĝNa))
for i = 1:length(ĝNa)
    println(ĝNa[i])
    ĝ = (ĝNa[i],36.,0.3)
    p̂ = (Iapp,c,ĝ,E)
    prob = ODEProblem(HH_ode!,x0,tspan,p̂)
    sôl = solve(prob,Tsit5(),saveat=dt,abstol=1e-9,reltol=1e-9,maxiters=1e6)
    v̂ = sôl[1,1,:]
    V[i] = sum((v-v̂).^2*dt)/Tfinal
end
dV = (V[2:end]-V[1:end-1])/dg

if save_data
    writedlm("./../data/HH_cost.txt",  hcat(ĝNa,V), " ")
    writedlm("./../data/HH_dcost.txt",  hcat(ĝNa[1:end-1],dV), " ")
end

# Plots
plt1=plot(ĝNa,V)
plt2=plot(ĝNa[1:end-1],dV)
ylims!((-50,50))
plot(plt1,plt2,layout=(2,1))

##
# initialize data matrix with true system i/o traces
data = hcat(t,Iapp.(t),v)

ĝNa_plot = [107,108]
plt = Plots.Plot{Plots.GRBackend}[]
plt2 = Plots.Plot{Plots.GRBackend}[]
for i = 1:2
    ĝ = (ĝNa_plot[i],36.,0.3)
    p̂ = (Iapp,c,ĝ,E)
    prob = ODEProblem(HH_ode!,x0,tspan,p̂)
    sôl = solve(prob,Tsit5(),saveat=dt,maxiters=1e6)
    v̂ = sôl[1,1,:]
    # add estimator i/o traces to data matrix
    data = hcat(data,v̂)
    V[i] = sum((v-v̂).^2)*dt/Tfinal
    
    push!(plt2,plot(t,(v-v̂).^2))
    plt_aux = plot(t,v)
    plt_aux = plot!(t,v̂,linecolor="red")
    push!(plt,plt_aux)
end
plot(plt[1],plt[2],layout=(2,1),legend=false)

if save_data
    writedlm("./../data/HH_spike.txt",  data, " ")
end