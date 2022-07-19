## Generate the data for the raster plots for the reliability experiment
# on Luka's bursting circuit.

# Reliability experiments on Luka's circuit

using Plots, Random, Distributions
using DifferentialEquations, LinearAlgebra, DSP, JLD

include("LR_odes.jl")
# Need to put a seed here?

num_trials = 4

max_error = 0.07 # 0.1 gives a mismatch of up to +/- 5%
max_tau_error = 0.02

d = Normal(0,1)
noise_sf = 20

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = -2
asp = 2
asn =  -1.5
ausp =  1.5

# If you change these or other deltas, remember to change 'delta_ests'.
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

# If you change this, make sure to also change 'delta_ests'
delta_ests_true = [dfn,dsp,dsn,dusp,delta_h,dfn,dsp,dsn,dusp,delta_h,deltasyn,deltasyn]


Tfinalrel = 20000;
dt = 0.1;

# Noise-generated current
noise = rand(d, round(Int, Tfinalrel/dt+1))*noise_sf;
Iconst = -2;

for i in eachindex(noise)
    i == 1 ? noise[i] = 0 : noise[i]=noise[i-1]+(noise[i]-noise[i-1])/2000
    # noise[i]=noise[i-1]*0.99974+noise[i]*(1-0.99974)
end
# # Filter the noise
# fs = 1e12
# responsetype = Lowpass(1; fs)
# designmethod = FIRWindow(hanning(1000; zerophase=false))
# #designmethod = FIRWindow(hanning(64; padding=30, zerophase=false))
# #designmethod = FIRWindow(rect(128; zerophase=false))
# #designmethod = Butterworth(4)
# fnoise = filt(digitalfilter(responsetype, designmethod), noise)

# noise = fnoise .+ Iconst;
noise = noise .+ Iconst;

Iapp2 = -0.65;

# Initialise arrays which will later be saved as .jld files.
delta_est_values = zeros(12,num_trials);
tau_est_values = zeros(4,num_trials);

Ref = zeros(2,Int(Tfinalrel/dt+2));
Mis = zeros(2,Int(Tfinalrel/dt+2),num_trials);
Learned = zeros(2,Int(Tfinalrel/dt+2),num_trials);

thetalearned = zeros(10,num_trials);

# Run the reference simulation.
Tfinal= Tfinalrel;
tspan=(0.0,Tfinal);
x0 = [-1.6 -1.6 -1.6 0 0 0];
u0 = x0;
p=(afn,asp,asn,ausp,afn2,asp2,asn2,ausp2,[tau_s tau_s],[tau_us tau_us],noise,Iapp2,asyn21,asyn12,delta_ests_true,beta)
probRef = ODEProblem(LR_ODE_rel_II!,u0,tspan,p) # Simulation without noise (ODE)
solRef = solve(probRef,Euler(),adaptive=false,dt=dt)
t = solRef.t;
Ref[1,:] = solRef[1,:];
Ref[2,:] = solRef[4,:];


p1=plot(t, Ref[1,:])

for idx in 1:num_trials
    println("Trial Number: $idx")
    mis = (rand(Uniform(0,max_error),12).-max_error/2).+1
    mis_tau = (rand(Uniform(0,max_tau_error),4).-max_tau_error/2).+1
    delta_ests = [0.01,-0.01,-0.88,-0.88,-0.5,-0.01,0.01,-0.88,-0.88,-0.5,-1,-1].*mis
    tau_ests = [tau_s, tau_us, tau_s, tau_us].*mis_tau

    global delta_est_values[:,idx] = delta_ests
    global tau_est_values[:,idx] = tau_ests
    println(tau_ests)

    # Simulate the mismatch neuron, before learning.
    Tfinal= Tfinalrel
    tspan=(0.0,Tfinal)
    x0 = [-1.5 -1.5 -1.5 -0.5 -0.5 -0.5]
    u0 = x0
    p=(afn,asp,asn,ausp,afn2,asp2,asn2,ausp2,[tau_ests[1] tau_ests[3]],
    [tau_ests[2] tau_ests[4]],noise,Iapp2,asyn21,asyn12,delta_ests,beta)
    probMis = ODEProblem(LR_ODE_rel_II!,u0,tspan,p)
    solMis = solve(probMis,Euler(),adaptive=false,dt=dt)
    global Mis[1,:,idx] = solMis[1,:]
    global Mis[2,:,idx] = solMis[4,:]

    # p1=plot(solRef.t, solRef[1,:])
    # plot!(solRef.t, solMis[1,:])


    # Now run the observer.
    Iappo = t -> -2 + 0.8*sin(0.001*t);
    # Iappo = -0.8;
    Iappo2 = -0.65;

    γ = 0.5;
    α = 0.0001;
    # Tfinal= 65000.0; # Non-diag observer converges faster.
    Tfinal= 60000.0;
    tspan=(0.0,Tfinal);

    x0 =  [-1.5 -1.5 -1.5 -0.5 -0.5 -0.5];
    xh0 = [-1 -1 -1 -1.5 -1.5 -1.5];
    θ̂₀ = 0.1*ones(10)
    P₀ = ones(10);
    Ψ₀ = zeros(10);
    u0 = [x0 xh0 θ̂₀' P₀' Ψ₀'];

    println("Learning...")
    p=(afn,asp,asn,ausp,afn2,asp2,asn2,ausp2,tau_s,tau_us,Iappo,Iappo2,
        asyn21,asyn12,delta_ests_true,delta_ests,beta,α,γ,tau_ests);
    probObs = ODEProblem(LR_observer_II!,u0,tspan,p) # Simulation without noise (ODE)
    solObs = solve(probObs,Euler(),adaptive=false,dt=dt)
    println("Finished learning.")

    # global plt0 = plot(solObs.t, solObs[1,:])
    # plot!(solObs.t, solObs[4,:])
    # global plt1 = plot(solObs.t, solObs[13,:])
    # plot!(solObs.t, solObs[17,:])
    # global plt2 = plot(solObs.t, solObs[14,:])
    # plot!(solObs.t, solObs[18,:])

    # Truncated figures
    j = size(solObs)[3]
    # i = round(Int,1*j/5)
    # plot(solObs.t[i:j], solObs[7,i:j])
    # plot!(solObs.t[i:j], -solObs[8,i:j]) # Plotting the negative so can compare!

    # Learned parameters
    afnl = mean(solObs[13,j-10000:j]);
    aspl = mean(solObs[14,j-10000:j]);
    asnl = mean(solObs[15,j-10000:j]);
    auspl = mean(solObs[16,j-10000:j]);
    asyn12l = mean(solObs[17,j-10000:j]);

    afn2l = mean(solObs[18,j-10000:j]);
    asp2l = mean(solObs[19,j-10000:j]);
    asn2l = mean(solObs[20,j-10000:j]);
    ausp2l = mean(solObs[21,j-10000:j]);
    asyn21l = mean(solObs[22,j-10000:j]);
    global thetalearned[:,idx] = [afnl aspl asnl auspl asyn12l afn2l asp2l asn2l ausp2l asyn21l];


    # Finally, apply the learned parameters!
    x0 = [-1.5 -1.5 -1.5 -0.5 -0.5 -0.5]
    u0 = x0
    Tfinal= Tfinalrel
    tspan=(0.0,Tfinal)
    p=(afnl,aspl,asnl,auspl,afn2l,asp2l,asn2l,ausp2l,[tau_ests[1] tau_ests[3]],
    [tau_ests[2] tau_ests[4]],noise,Iapp2,asyn21l,asyn12l,delta_ests,beta)
    probLearned = ODEProblem(LR_ODE_rel_II!,u0,tspan,p)  # Simulation without noise (ODE)
    solLearned = solve(probLearned,Euler(),adaptive=false,dt=dt)
    global Learned[1,:,idx] = solLearned[1,:]
    global Learned[2,:,idx] = solLearned[4,:]

    # p2=plot(solRef.t, solRef[1,:])
    # plot!(solLearned.t, solLearned[1,:])
end

save("sec4_LR_II_tautest.jld","noise",noise,
    "delta_est_values",delta_est_values,"tau_est_values",tau_est_values,
    "thetalearned",thetalearned,
    "t",t,"Ref",Ref,"Mis",Mis,"Learned",Learned)