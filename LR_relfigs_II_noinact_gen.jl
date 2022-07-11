## Generate the data for the raster plots for the reliability experiment
# on Luka's bursting circuit.

# Reliability experiments on Luka's circuit

using Plots, Random, Distributions
using DifferentialEquations, LinearAlgebra, DSP, JLD

include("LR_odes.jl")
# Need to put a seed here?

num_trials = 6

max_error = 0.1 # 0.08 # 0.1 gives a mismatch of up to +/- 5%
max_tau_error = 0.00001 # 0.02

d = Normal(0,1)
noise_sf = 18

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = -2
asp = 2
asn =  -1.5
ausp =  1.5

# If you change these or other deltas, remember to change 'delta_ests',
# as it's hardcoded !!
dfn = 0
dsp = 0
dsn = -0.9
dusp = -2.8

afn2 = -2
asp2 = 2
asn2 =  -1.5
ausp2 =  1.5

asyn21 = -2
asyn12 = -2

deltasyn = -1
delta_h = -0.5
beta = 2

# If you change this, make sure to also change 'delta_ests'
delta_ests_true = [dfn,dsp,dsn,dusp,delta_h,dfn,dsp,dsn,dusp,delta_h,deltasyn,deltasyn]


Tfinalrel = 25000;
dt = 0.1;

# Noise-generated current
noise = rand(d, round(Int, Tfinalrel/dt+1))*noise_sf;
Iconst = 0.;

for i in eachindex(noise)
    i == 1 ? noise[i] = 0 : noise[i]=noise[i-1]+(noise[i]-noise[i-1])/4000
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

Iapp2 = 0.;

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
probRef = ODEProblem(LR_ODE_rel_II_noinact!,u0,tspan,p) # Simulation without noise (ODE)
solRef = solve(probRef,Euler(),adaptive=false,dt=dt)
t = solRef.t;
Ref[1,:] = solRef[1,:];
Ref[2,:] = solRef[4,:];

pRef=plot(t, Ref[1,:])
plot!(t, Ref[2,:])

for idx in 1:num_trials
    println("Trial Number: $idx")
    mis = (rand(Uniform(0,max_error),12).-max_error/2).+1
    mis_tau = (rand(Uniform(0,max_tau_error),4).-max_tau_error/2).+1
    delta_ests = [0.01,-0.01,-0.9,-2.8,-0.5,-0.01,0.01,-0.9,-2.8,-0.5,-1,-1].*mis
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
    probMis = ODEProblem(LR_ODE_rel_II_noinact!,u0,tspan,p)
    solMis = solve(probMis,Euler(),adaptive=false,dt=dt)
    global Mis[1,:,idx] = solMis[1,:]
    global Mis[2,:,idx] = solMis[4,:]

    global pCompBef=plot(solRef.t, solRef[1,:]) # Compare before
    plot!(solRef.t, solMis[1,:])


    # Now run the observer.
    # Iappo = t -> 0. + 0.4*sin(0.0008*t); # 0.2 + 0.4*sin(0.001*t);
    function step(t)
        0.5 * (sign(t) + 1)
    end
    function interval(t, a, b)
        heaviside(t-a) - heaviside(t-b)
    end
    function piecewise(t)
        out = 0.
        num_pulses = 80
        pulsetimes = zeros(num_pulses)
        for i=1:num_pulses
            pulsetimes[i] = 5000 + 4500*(i-1)
        end
        for i=1:num_pulses
            out = out - 0.6*interval(t, pulsetimes[i], pulsetimes[i]+2000)
        end
        out
    end

    Iappo = t -> piecewise(t)
    
    # Iappo = -0.8;
    Iappo2 = 0.;

    γ = 0.2;
    α = 0.0001;
    # Tfinal= 65000.0; # Non-diag observer converges faster.
    Tfinal= 60000; # 290000; # 300000.0;
    tspan=(0.0,Tfinal);

    x0 =  [-1.5 -1.5 -1.5 -0.5 -0.5 -0.5];
    xh0 = [-1 -1 -1 -1 -1 -1];
    θ̂₀ = [-1 1 -1 1 -1 -1 1 -1 1 -1]'; # 0.1*ones(10)
    # P₀ = ones(10);
    P₀ = Matrix(I, 5, 5);
    Ψ₀ = zeros(10);
    # u0 = [x0 xh0 θ̂₀' P₀' Ψ₀']; # For diag observer.
    u0 = [x0 xh0 θ̂₀' P₀[:]' P₀[:]' Ψ₀'];

    println("Learning...")
    p=(afn,asp,asn,ausp,afn2,asp2,asn2,ausp2,tau_s,tau_us,Iappo,Iappo2,
        asyn21,asyn12,delta_ests_true,delta_ests,beta,α,γ,tau_ests);
    probObs = ODEProblem(LR_observer_II_noinact_nondiag!,u0,tspan,p) # Simulation without noise (ODE)
    # global solObs = solve(probObs,Euler(),adaptive=false,dt=dt)
    global solObs = solve(probObs,Euler(),adaptive=false,dt=dt)
    println("Finished learning.")

    global plt0 = plot(solObs.t, solObs[1,:])
    plot!(solObs.t, solObs[4,:])
    global plt1 = plot(solObs.t, solObs[13,:])
    plot!(solObs.t, solObs[18,:])
    global plt2 = plot(solObs.t, solObs[14,:])
    plot!(solObs.t, solObs[19,:])

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

    global pCompAfter=plot(solRef.t, solRef[1,:])
    plot!(solLearned.t, solLearned[1,:])
end

save("sec4_LR_II_noinact.jld","noise",noise,
    "delta_est_values",delta_est_values,"tau_est_values",tau_est_values,
    "thetalearned",thetalearned,
    "t",t,"Ref",Ref,"Mis",Mis,"Learned",Learned)
