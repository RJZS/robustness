using Plots: print
using DifferentialEquations, Random, Plots, LinearAlgebra, DelimitedFiles

# Flag for saving data to .txt files 
save_data = false

# No internal dynamics perturbation
δ = ones(46,4)

# Models (include after defining the perturbation δ)
include("./HCO_kinetics.jl")
include("./HCO_odes.jl")

# Definition of the timespan for numerical integration
Tfinal = 15000
tspan = (0.0,Tfinal)

# Input currents:
Iapp1 = t -> -0.65  
Iapp2 = t -> -0.65 

# Nominal parameters
C1 = C2 = 1.0
(gNa1,gNa2) = (60.,60.)                 # Sodium current maximal conductance
(gK1,gK2) = (40.,40.)                   # Potassium current maximal conductance
(gL1,gL2) = (0.035,0.035)               # Leak maximal conductance
(gSyn1,gSyn2) = (4,4)                   # Synaptic maximal conductance

# Calcium current maximal conductance
gCa0 = 0.1
gCaf = 0.16
sig = x -> 1 ./ (1+exp(-x));
gCa = t -> gCa0 .+ (gCaf-gCa0)*sig((t-Tfinal/2)/1250);#750
(gCa1,gCa2) = (t -> gCa.(t), t -> gCa.(t))     

# Synaptic parameters
a1 = a2 = 2;        
b1 = b2 = 0.1;

# Observer parameters
α₁ = 0.01
α₂ = 1.
γ = 10.
β = γ

# True system initial conditions (chosen from steady
# state oscillations with gCa = 0.1)
V0 =  [ -62.66091038531592
        -70.44894441288831
        ]
w₀ =  [ 0.0058621887308486515
        0.001344578028506826
        0.9340962269868666
        0.984775915082517
        0.014066710466720091
        0.0068454789625906455
        0.6713459367621024
        0.3257086515030068
        0.025756012084214033
        0.19423394334660857
        4.0971753762831785e-5
        0.0035516641693319776
        ]
x0 = [V0;w₀]
# V01= -64
# V02= -66.
# x01 = [V01 0.5*ones(1,6)]   #gamNa_m(V01,1) gamNa_h(V01,1) gamK_m(V01,1) gamCa_m(V01,1) gamCa_h(V01,1) gamSyn(V01,1)]
# x02 = [V02 0.5*ones(1,6)]   #gamNa_m(V02,2) gamNa_h(V02,2) gamK_m(V02,2) gamCa_m(V02,2) gamCa_h(V02,2) gamSyn(V02,2)]
# x0 = reshape([x01;x02],1,14)'
# Observer initial conditions
x̂01 = [-50. zeros(1,6)]
x̂02 = [-50. zeros(1,6)]
x̂0 = reshape([x̂01;x̂02],1,14)'
gNa1₀ = gNa2₀ = 80
gK1₀ = gK2₀ = 80
gCa1₀ = gCa2₀ = 1 
gL1₀ = gL2₀ = 1
gSyn1₀ = gSyn2₀ = 10
θ₀ = reshape([gNa1₀ gK1₀ gCa1₀ gL1₀ gSyn1₀; gNa2₀ gK2₀ gCa2₀ gL2₀ gSyn2₀],10,1)
P0 = (Matrix(I, 5, 5),Matrix(I, 5, 5))
ψ₀ = zeros(10,1)
z0 = [x0;x̂0;θ₀;reshape(P0[1],25,1);reshape(P0[2],25,1);ψ₀]

## ESTIMATE BOTH TRUE SYSTEM AND OBSERVER
p = ((Iapp1,Iapp2),(C1,C2),(gNa1,gNa2),(gK1,gK2),(gCa1,gCa2),(gL1,gL2),(gSyn1,gSyn2),
    (a1,a2),(b1,b2),(α₁,α₂,β,γ))

prob = ODEProblem(HCO_observer_ode,z0,tspan,p)
sol = solve(prob,Tsit5(),saveat=0.1,reltol=1e-5,abstol=1e-5) #,reltol=1e-6,abstol=1e-6,AutoTsit5(Rosenbrock23()),Tsit5()
v = sol[1:2,1,:]';
w = sol[3:14,1,:]';
v₀ = v[end,:] 
w₀ = w[end,:] 
v̂ = sol[15:16,1,:]';
θ = sol[29:38,1,:]';

## PLOTS

# VOLTAGES
p1=plot(sol.t, v[:,1],linewidth=1.5,legend=false)
p2=plot(sol.t, v[:,2],linewidth=1.5,legend=false)
voltages = plot(p1,p2,layout=(2,1),legend=false)

# Na CONDUCTANCES 
plt1 = plot(sol.t,θ[:,1:2],labels=["ĝNa₁" "ĝNa₂"],linecolor=["red" "blue"])
plt1 = plot!([0,Tfinal],[gNa1,gNa1],linecolor="red",linestyle= :dash,labels="gNa₁")
plt1 = plot!([0,Tfinal],[gNa2,gNa2],linecolor="blue",linestyle= :dash,labels="gNa₂")

# K CONDUCTANCES 
plt2 = plot(sol.t,θ[:,3:4],labels=["ĝK₁" "ĝK₂"],linecolor=["red" "blue"]) 
plt2 = plot!([0,Tfinal],[gK1,gK1],linecolor="red",linestyle= :dash,labels="gK₁")
plt2 = plot!([0,Tfinal],[gK2,gK2],linecolor="blue",linestyle= :dash,labels="gK₂")

# Ca CONDUCTANCES 
plt3 = plot(sol.t,θ[:,5:6],labels=["ĝCa₁" "ĝCa₂"],linecolor=["red" "blue"])
plt3 = plot!(sol.t,[gCa1(sol.t)],linecolor="red",linestyle= :dash,labels="gCa₁")
plt3 = plot!(sol.t,[gCa2(sol.t)],linecolor="blue",linestyle= :dash,labels="gCa₂")
ylims!((-0.1,0.2))

# Leak CONDUCTANCES
plt4 = plot(sol.t,θ[:,7:8],labels=["ĝL₁" "ĝL₂"],linecolor=["red" "blue"]) 
plt4 = plot!([0,Tfinal],[gL1,gL1],linecolor="red",linestyle= :dash,labels="gL₁")
plt4 = plot!([0,Tfinal],[gL2,gL2],linecolor="blue",linestyle= :dash,labels="gL₂")

# # SYNAPTIC CONDUCTANCES
plt5 = plot(sol.t,θ[:,9:10],labels=["ĝSyn₁" "ĝSyn₂"],linecolor=["red" "blue"])
plt5 = plot!([0,Tfinal],[gSyn1,gSyn1],linecolor="red",linestyle= :dash,labels="gSyn₁")
plt5 = plot!([0,Tfinal],[gSyn2,gSyn2],linecolor="blue",linestyle= :dash,labels="gSyn₂")

if save_data
    writedlm("./../data/HCO_truevoltages_neuromodulation.txt",  hcat(sol.t,v), " ")
    writedlm("./../data/HCO_parameters_neuromodulation.txt",  hcat(sol.t,θ,gCa.(sol.t)), " ")
end
plot(plt1,plt2,plt5,layout=(3,1),legend=true)
plot(plt3,plt4,layout=(2,1),legend=true)