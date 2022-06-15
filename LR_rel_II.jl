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
ausp =  1.5

dfn = 0
dsp = 0
dsn = -0.88
dusp = -0.88

afn2 = -2
asp2 = 2
asn2 =  -1.7
ausp2 =  1.5

asyn21 = -0.2
asyn12 = -0.2

deltasyn = -1
delta_h = -0.5
beta = 2

# delta_ests = (0, 0, -1.5, -1.5)
# delta_ests = (.4, -.3, -1.2, -1.9)

delta_ests_true = [dfn,dsp,dsn,dusp,delta_h,dfn,dsp,dsn,dusp,delta_h,deltasyn,deltasyn]
mis = (rand(Uniform(0,0.1),12).-0.05).+1
delta_ests = [0.01,-0.01,-0.88,-0.88,-0.5,-0.01,0.01,-0.88,-0.88,-0.5,-1,-1].*mis

# Initial conditions
x0 = [-1.6 -1.6 -1.6 0 0 0]
u0 = x0

dt = 0.1

# Noise-generated current
d = Normal(0,1)
Tfinalrel = 20000
noise = rand(d, round(Int, Tfinalrel/dt+1))*15
Iconst = -2

for i in eachindex(noise)
    i == 1 ? noise[i] = 0 : noise[i]=noise[i-1]+(noise[i]-noise[i-1])/2000
end
noise = noise .+ Iconst
plot(noise)

Iapp2 = -0.65

save("LR_rel_II.jld","noise",noise,"mis",mis)
# noise = load("LR_rel_II.jld")["noise"]
# mis = load("LR_rel_II.jld")["mis]


# Simulation
Tfinal= Tfinalrel
tspan=(0.0,Tfinal)
p=(afn,asp,asn,ausp,afn2,asp2,asn2,ausp2,tau_s,tau_us,noise,Iapp2,asyn21,asyn12,delta_ests_true,beta)
probRef = ODEProblem(LR_ODE_rel_II!,u0,tspan,p) # Simulation without noise (ODE)
solRef = solve(probRef,Euler(),adaptive=false,dt=dt)

plot(solRef.t, solRef[1,:],linewidth=1.5,legend=false)

# p1=plot(solRef.t, solRef[1,:],linewidth=1.5,legend=false)
# ylabel!("V")

# Now try the mismatched neuron.
x0 = [-1.5 -1.5 -1.5 -0.5 -0.5 -0.5]
u0 = x0
Tfinal= Tfinalrel
tspan=(0.0,Tfinal)
p=(afn,asp,asn,ausp,afn2,asp2,asn2,ausp2,tau_s,tau_us,noise,Iapp2,asyn21,asyn12,delta_ests,beta)
probMis = ODEProblem(LR_ODE_rel_II!,u0,tspan,p) # Simulation without noise (ODE)
solMis = solve(probMis,Euler(),adaptive=false,dt=dt)

p1=plot(solRef.t, solRef[1,:])
plot!(solRef.t, solMis[1,:])


# Now run the observer.
Iappobs = -2;

γ =0.01;
α = 0.0001;
Tfinal= 60000.0;
tspan=(0.0,Tfinal);

x0 = [-1.9 -1.9 -1.9];
xh0 = [-1 -1 -1];
θ̂₀ = [.1 .1 .1 .1];
P₀ = [1 1 1 1];
Ψ₀ = [0 0 0 0];
u0 = [x0 xh0 θ̂₀ P₀ Ψ₀];

p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iappobs,delta_ests);
probObs = ODEProblem(LR_observer_noinact!,u0,tspan,p) # Simulation without noise (ODE)
solObs = solve(probObs,Euler(),adaptive=false,dt=dt)


# Truncated figures
j = size(solObs)[3]
i = round(Int,1*j/5)
plot(solObs.t[i:j], solObs[7,i:j])
plot!(solObs.t[i:j], -solObs[8,i:j]) # Plotting the negative so can compare!

# Learned parameters
afnl = mean(solObs[7,j-10000:j]);
aspl = mean(solObs[8,j-10000:j]);
asnl = mean(solObs[9,j-10000:j]);
auspl = mean(solObs[10,j-10000:j]);


# Finally, apply the learned parameters!
x0 = [-1.5 -1.5 -1.5]
u0 = x0
Tfinal= Tfinalrel
tspan=(0.0,Tfinal)
p=(afnl,aspl,asnl,auspl,dfn,dsp,dsn,dusp,tau_s,tau_us,noise,delta_ests)
probLearned = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solLearned = solve(probLearned,Euler(),adaptive=false,dt=dt)

p2=plot(solRef.t, solRef[1,:])
plot!(solLearned.t, solLearned[1,:])
