## Generate the data for the raster plots for the reliability experiment
# on Luka's bursting circuit.

# Reliability experiments on Luka's circuit

using Plots, Random, Distributions
using DifferentialEquations, LinearAlgebra, JLD, FFTW

# There's a useful guide to using FFTW here:
# https://www.matecdev.com/posts/julia-fft.html

include("../LR_odes.jl")
# Need to put a seed here?

num_trials = 1

max_error = 0.1 # 0.1 gives a mismatch of up to +/- 5%
max_tau_error = 0.04

d = Normal(0,1)
noise_sf = 12

## Definition of parameters
tau_s = 50
tau_us = 50*50

afn = -2
asp = 2
asn =  -1.5
ausp =  2

# NOTE: These are not directly used by LR_ODE_rel!
# If you change these, remember to change 'delta_ests'.
dfn = 0
dsp = 0
dsn = -1.5
dusp = -1.5

# If you change this, make sure to also change 'delta_ests'
delta_ests_true = [dfn,dsp,dsn,dusp]

delta_h = -0.5
beta = 2


Tfinalrel = 20000
dt = 0.1

# Noise-generated current
noise = rand(d, round(Int, Tfinalrel/dt+1))*noise_sf
Iconst = -2.6

for i in eachindex(noise)
    i == 1 ? noise[i] = 0 : noise[i]=noise[i-1]+(noise[i]-noise[i-1])/2000
end
noise = noise .+ Iconst

# Initialise arrays which will later be saved as .jld files.
delta_est_values = zeros(4,num_trials)
tau_est_values = zeros(2,num_trials)

Ref = zeros(Int(Tfinalrel/dt+2))
Mis = zeros(Int(Tfinalrel/dt+2),num_trials)
Learned = zeros(Int(Tfinalrel/dt+2),num_trials)

RefStep = zeros(Int(Tfinalrel/dt+2))
MisStep = zeros(Int(Tfinalrel/dt+2),num_trials)
LearnedStep = zeros(Int(Tfinalrel/dt+2),num_trials)

thetalearned = zeros(4,num_trials)
thetalearnedStep = zeros(4,num_trials)

# Run the reference simulation.
Tfinal= Tfinalrel
tspan=(0.0,Tfinal)
x0 = [-1.8 -1.8 -1.8]
u0 = x0
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,noise,delta_ests_true)
probRef = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solRef = solve(probRef,Euler(),adaptive=false,dt=dt)
t = solRef.t
Ref = solRef[1,:]

# Now the step version
Istep = -2.2*ones(Int(Tfinalrel/dt+1))
Istep[1:30000] .= -2.6
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Istep,delta_ests_true)
probRefStep = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solRefStep = solve(probRefStep,Euler(),adaptive=false,dt=dt)
RefStep = solRefStep[1,:]

idx = 1
println("Trial Number: $idx")
mis = (rand(Uniform(0,max_error),4).-max_error/2).+1
mis_tau = (rand(Uniform(0,max_tau_error),2).-max_tau_error/2).+1
delta_ests = [0.01,0.01,-1.5,-1.5].*mis
tau_ests = [tau_s, tau_us].*mis_tau
println(tau_ests)

global delta_est_values[:,idx] = delta_ests
global tau_est_values[:,idx] = tau_ests

# Simulate the mismatch neuron, before learning.
Tfinal= Tfinalrel
tspan=(0.0,Tfinal)
x0 = [-1.5 -1.5 -1.5]
u0 = x0
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_ests[1],tau_ests[2],noise,delta_ests)
probMis = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solMis = solve(probMis,Euler(),adaptive=false,dt=dt)
global Mis[:,idx] = solMis[1,:]

# Now the version with the step
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_ests[1],tau_ests[2],Istep,delta_ests)
probMisStep = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solMisStep = solve(probMisStep,Euler(),adaptive=false,dt=dt)
global MisStep[:,idx] = solMisStep[1,:]

# p1=plot(solRef.t, solRef[1,:])
# plot!(solRef.t, solMis[1,:])


# Now run the observer.
Iappobs = t -> -2.6 + 0.4*sin(0.001*t)
IappobsStep = t -> -2.2

γ = 2;
α = 0.0001;
# Tfinal= 65000.0; # Non-diag observer converges faster.
Tfinal= 60000.0;
tspan=(0.0,Tfinal);

x0 = [-1.9 -1.9 -1.9];
xh0 = [-1 -1 -1];
θ̂₀ = [.1 .1 .1 .1];
P₀ = [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1]; # For non-diag observer.
# P₀ = [1 1 1 1];
Ψ₀ = [0 0 0 0];
u0 = [x0 xh0 θ̂₀ P₀ Ψ₀];

println("Learning...")
p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,Iappobs,delta_ests,α,γ,tau_ests);
probObs = ODEProblem(LR_observer_noinact_nondiag!,u0,tspan,p) # Simulation without noise (ODE)
solObs = solve(probObs,Euler(),adaptive=false,dt=dt)

p=(afn,asp,asn,ausp,dfn,dsp,dsn,dusp,tau_s,tau_us,IappobsStep,delta_ests,α,γ,tau_ests);
probObsStep = ODEProblem(LR_observer_noinact_nondiag!,u0,tspan,p) # Simulation without noise (ODE)
global solObsStep = solve(probObsStep,Euler(),adaptive=false,dt=dt)
println("Finished learning.")

# global plt0 = plot(solObs.t, solObs[1,:])
# global plt1 = plot(solObs.t, solObs[7,:])
# plot!(solObs.t, -solObs[8,:])
# global plt2 = plot(solObs.t, solObs[9,:])

# Truncated figures
j = size(solObs)[3]
# i = round(Int,1*j/5)
# plot(solObs.t[i:j], solObs[7,i:j])
# plot!(solObs.t[i:j], -solObs[8,i:j]) # Plotting the negative so can compare!

# Learned parameters
afnl = mean(solObs[7,j-10000:j]);
aspl = mean(solObs[8,j-10000:j]);
asnl = mean(solObs[9,j-10000:j]);
auspl = mean(solObs[10,j-10000:j]);
global thetalearned[:,idx] = [afnl aspl asnl auspl];

# And for the step
afnlS = mean(solObsStep[7,j-10000:j]);
asplS = mean(solObsStep[8,j-10000:j]);
asnlS = mean(solObsStep[9,j-10000:j]);
ausplS = mean(solObsStep[10,j-10000:j]);
global thetalearnedStep[:,idx] = [afnlS asplS asnlS ausplS];


# Finally, apply the learned parameters!
x0 = [-1.5 -1.5 -1.5]
u0 = x0
Tfinal= Tfinalrel
tspan=(0.0,Tfinal)
p=(afnl,aspl,asnl,auspl,dfn,dsp,dsn,dusp,tau_ests[1],tau_ests[2],noise,delta_ests)
probLearned = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solLearned = solve(probLearned,Euler(),adaptive=false,dt=dt)
global Learned[:,idx] = solLearned[1,:]

# And for the step
p=(afnlS,asplS,asnlS,ausplS,dfn,dsp,dsn,dusp,tau_ests[1],tau_ests[2],Istep,delta_ests)
probLearnedStep = ODEProblem(LR_ODE_rel!,u0,tspan,p) # Simulation without noise (ODE)
solLearnedStep = solve(probLearnedStep,Euler(),adaptive=false,dt=dt)
global LearnedStep[:,idx] = solLearnedStep[1,:]

# p2=plot(solRef.t, solRef[1,:])
# plot!(solLearned.t, solLearned[1,:])

t = solLearned.t;
y = solLearned[1,:];
plot(t,y)

z = y .- mean(y); # Removing DC component.

x = solLearnedStep[1,:];
xz = x .- mean(x);

fs = Int(1/dt); # Sampling rate (Hz)
# F = fftshift(fft(y));
# freqs = fftshift(fftfreq(length(t), fs));
# Since the signal is real, can use the functions below instead. They're faster.
# No fftshift is needed, as only positive frequencies are generated.
# Recall that for real signals, F(-w) = F(w).
F = rfft(z);
Fx = rfft(xz); # For the step input simulation.
freqs = rfftfreq(length(t),fs);

j = 500
aF = abs.(F)
aFx = abs.(Fx);
plot(freqs[1:j], aF[1:j])

# Note tau_s = 50, tau_us = 50*50
# Those have inverses of 0.02, and 0.0004.

# 16/06: Used bilinear transform to approximate the filter s*gamma/(s+gamma)
yHi = copy(y) # High pass filtered y.
for i in eachindex(y)
    i < 3 ? yHi[i] = y[i] : 
            yHi[i] = (-(γ-1)/(1+γ))*yHi[i-1] + (1/(1+γ))*(y[i] - y[i-1])
end

plot(t[6:200002],yHi[6:200002]) # Why large transients at the start?
plot(t[15000:50002],yHi[15000:50002]) # Zoomed in.
FHi = rfft(yHi .- mean(yHi))
aFHi = abs.(FHi)
plot(freqs, aFHi)
plot(freqs[1:8000], aFHi[1:8000])

## Testing Code
yTest = sin.(4*pi*t);
Ftest = rfft(yTest);
plot(freqs,abs.(Ftest))

yHiT = copy(yTest) # High pass filtered yTest.
for i in eachindex(yTest)
    i < 3 ? yHiT[i] = yTest[i] : 
            yHiT[i] = (-(γ-1)/(1+γ))*yHiT[i-1] + (1/(1+γ))*(yTest[i] - yTest[i-1])
end

# Now apply the bandpass filter
γ = 0.00001*2*pi;
β = 0.005*2*pi;
yBand = copy(y);
for k in eachindex(y)
    if k == 1
        yBand[k] = y[k]
    elseif k == 2
        yBand[k] = -( (γ-1)/(1+γ) + (β-1)/(1+β) )*yBand[k-1] + 
                    y[k]*β/((1+γ)*(1+β))
    else
        yBand[k] = -( (γ-1)/(1+γ) + (β-1)/(1+β) )*yBand[k-1] + 
                    -yBand[k-2]*((γ-1)*(β-1))/((1+γ)*(1+β)) +
                    (y[k]-y[k-2])*β/((1+γ)*(1+β))
    end
end
plot(t[50002:100002],yBand[50002:100002]) # Why large transients at the start?
FBand = rfft(yBand); # Need to subtract off mean?
aFBand = abs.(FBand);
plot(freqs[1:2000], aFBand[1:2000])

# Much more effective to use a standard digital filter.
filter1=DSP.Filters.Chebyshev1(4,2)
thefilter = DSP.Filters.digitalfilter(Bandpass(0.00001,0.01;fs),filter1)
yFiltered = DSP.filt(thefilter, y)
plot(t,yFiltered)