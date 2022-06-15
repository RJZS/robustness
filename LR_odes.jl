# The model introduced in "Neuromodulation of Neuromorphic Circuits"
# by Luka Ribar.

using DifferentialEquations

function sigmoid(x, k=1)
    1 ./ (1 .+ exp(-k * x))
end

function element(V, a, delta)
    a * tanh(V - delta)
end

function synapse(V, a, delta, beta)
    a * sigmoid(V-delta, beta)
end

function LR_ODE!(du,u,p,t)
    # Gains
    afn = p[1]
    asp = p[2]
    asn = p[3]
    ausp = p[4]

    # Offsets
    dfn = p[5]
    dsp = p[6]
    dsn = p[7]
    dusp = p[8]

    # Time constants
    tau_s = p[9]
    tau_us = p[10]

    # Input
    Iapp = p[11]
    I1=p[12] # Amplitude of first step input
    I2=p[13] # Amplitude of second step input
    ti1=p[14] # Starting time of first step input
    tf1=p[15] # Ending time of first step input
    ti2=p[16] # Starting time of second step input
    tf2=p[17] # Ending time of second step input

    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    # ODEs
    du[1] = -V  -element(V,afn,dfn) -element(Vs,asp,dsp) +
                -element(Vs,asn,dsn) -element(Vus,ausp,dusp) +
                Iapp + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2)
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)
end

function LR_II_ODE!(du,u,p,t)
    # Gains
    afn = p[1]
    asp = p[2]
    asn = p[3]
    ausp = p[4]

    # Offsets
    dfn = p[5]
    dsp = p[6]
    dsn = p[7]
    dusp = p[8]

    # Time constants
    tau_s = p[9]
    tau_us = p[10]

    # Input
    Iapp = p[11]
    I1=p[12] # Amplitude of first step input
    I2=p[13] # Amplitude of second step input
    ti1=p[14] # Starting time of first step input
    tf1=p[15] # Ending time of first step input
    ti2=p[16] # Starting time of second step input
    tf2=p[17] # Ending time of second step input

    # Gains of second neuron
    afn2 = p[18]
    asp2 = p[19]
    asn2 = p[20]
    ausp2 = p[21]

    # Input to second neuron
    Iapp2 = p[22]

    # Synaptic parameters
    asyn21 = p[23]
    asyn12 = p[24]

    deltasyn = p[25]
    beta = 2

    I3 = p[26]
    ti3 = p[27]
    tf3 = p[28]

    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    V2 = u[4]
    Vs2 = u[5]
    Vus2 = u[6]

    # ODEs
    du[1] = -V  -element(V,afn,dfn) -element(Vs,asp,dsp) +
                -element(Vs,asn,dsn) -element(Vus,ausp,dusp) +
                synapse(Vs2, asyn12, deltasyn, beta) +
                Iapp + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2)
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)

    du[4] = -V2  -element(V2,afn2,dfn) -element(Vs2,asp2,dsp) +
                -element(Vs2,asn2,dsn) -element(Vus2,ausp2,dusp) +
                synapse(Vs, asyn21, deltasyn, beta) +
                Iapp2 + I3*pulse(t,ti3,tf3)
    du[5] = (1/tau_s)  * (V2 - Vs2)
    du[6] = (1/tau_us) * (V2 - Vus2)
end

function LR_II_ODE_with_inact!(du,u,p,t)
    # Gains
    afn = p[1]
    asp = p[2]
    asn = p[3]
    ausp = p[4]

    # Offsets
    dfn = p[5]
    dsp = p[6]
    dsn = p[7]
    dusp = p[8]

    # Time constants
    tau_s = p[9]
    tau_us = p[10]

    # Input
    Iapp = p[11]
    I1=p[12] # Amplitude of first step input
    I2=p[13] # Amplitude of second step input
    ti1=p[14] # Starting time of first step input
    tf1=p[15] # Ending time of first step input
    ti2=p[16] # Starting time of second step input
    tf2=p[17] # Ending time of second step input

    # Gains of second neuron
    afn2 = p[18]
    asp2 = p[19]
    asn2 = p[20]
    ausp2 = p[21]

    # Input to second neuron
    Iapp2 = p[22]

    # Synaptic parameters
    asyn21 = p[23]
    asyn12 = p[24]

    deltasyn = p[25]
    beta = 2

    I3 = p[26]
    ti3 = p[27]
    tf3 = p[28]

    delta_h = p[29]

    I4 = p[30]
    ti4 = p[31]
    tf4 = p[32]

    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    V2 = u[4]
    Vs2 = u[5]
    Vus2 = u[6]

    # ODEs
    asn_with_inact = asn * sigmoid(-(Vus-delta_h),beta)
    du[1] = -V  -element(V,afn,dfn) -element(Vs,asp,dsp) +
                -element(Vs,asn_with_inact,dsn) -element(Vus,ausp,dusp) +
                synapse(Vs2, asyn12, deltasyn, beta) +
                Iapp + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2)
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)

    asn2_with_inact = asn2 * sigmoid(-(Vus2-delta_h),beta)
    du[4] = -V2  -element(V2,afn2,dfn) -element(Vs2,asp2,dsp) +
                -element(Vs2,asn2_with_inact,dsn) -element(Vus2,ausp2,dusp) +
                synapse(Vs, asyn21, deltasyn, beta) +
                Iapp2 + I3*pulse(t,ti3,tf3) + I4*pulse(t,ti4,tf4)
    du[5] = (1/tau_s)  * (V2 - Vs2)
    du[6] = (1/tau_us) * (V2 - Vus2)
end

# For running the reliability experiment.
function LR_ODE_rel!(du,u,p,t)
    # Gains
    afn = p[1]
    asp = p[2]
    asn = p[3]
    ausp = p[4]

    # Offsets
    dfn = p[5]
    dsp = p[6]
    dsn = p[7]
    dusp = p[8]

    # Time constants
    tau_s = p[9]
    tau_us = p[10]

    # Input
    noise = p[11]

    delta_ests = p[12]

    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    # ODEs
    du[1] = -V  -element(V,afn,delta_ests[1]) -element(Vs,asp,delta_ests[2]) +
                -element(Vs,asn,delta_ests[3]) -element(Vus,ausp,delta_ests[4]) +
                noise[round(Int, t/dt)+1]
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)
end

# Original, v observer.
function LR_observer_noinact!(du,u,p,t)
    # Gains
    afn = p[1]
    asp = p[2]
    asn = p[3]
    ausp = p[4]
    # Offsets
    dfn = p[5]
    dsp = p[6]
    dsn = p[7]
    dusp = p[8]
    # Time constants
    tau_s = p[9]
    tau_us = p[10]
    # Input
    Iapp = p[11]
    delta_ests = p[12] # Estimated deltas (for uncertain model)
    
    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    # ODEs
    du[1] = -V  -element(V,afn,dfn) -element(Vs,asp,dsp) +
                -element(Vs,asn,dsn) -element(Vus,ausp,dusp) +
                Iapp
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)

    # Adaptive Observer
    Vh = u[4]
    Vsh = u[5]
    Vush = u[6]
    θ̂= u[7:10]
    # P = reshape(u[28+1:28+4],2,2);    
    # P = (P+P')/2
    P = u[11:14]
    Ψ = u[15:18]

    ϕ̂ = [-element(V,1,delta_ests[1]) ...
        -element(Vsh,1,delta_ests[2]) ...
        -element(Vsh,1,delta_ests[3]) ...
        -element(Vush,1,delta_ests[4])]

    du[4] = dot(ϕ̂,θ̂) -V  + Iapp + γ*(1+sum(P.*Ψ.^2))*(V-Vh) # γ*(1+Ψ'*P*Ψ)*(V-Vh)

    du[5] = (1/tau_s) * (V - Vsh)
    du[6] = (1/tau_us) * (V - Vush)
    
    du[7:10]= γ*P.*Ψ*(V-Vh); # dθ̂ 
    du[15:18] = -γ*Ψ + ϕ̂;  # dΨ
    du[11:14] = α*P - α*P.^2 .* Ψ.^2; # ((P*Ψ)*(P*Ψ)');
    # dP = (dP+dP')/2;
    # du[8] = dP[:]
end

# v observer with inactivation
function LR_observer!(du,u,p,t)
    # Gains
    afn = p[1]
    asp = p[2]
    asn = p[3]
    ausp = p[4]
    # Offsets
    dfn = p[5]
    dsp = p[6]
    dsn = p[7]
    dusp = p[8]
    # Time constants
    tau_s = p[9]
    tau_us = p[10]
    # Input
    Iapp = p[11]
    delta_ests = p[12] # Estimated deltas (for uncertain model)
    delta_h = p[13]
    beta = p[14]
    
    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    # ODEs
    asn_with_inact = asn * sigmoid(-(Vus-delta_h),beta)
    du[1] = -V  -element(V,afn,dfn) -element(Vs,asp,dsp) +
                -element(Vs,asn_with_inact,dsn) -element(Vus,ausp,dusp) +
                Iapp
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)

    # Adaptive Observer
    Vh = u[4]
    Vsh = u[5]
    Vush = u[6]
    θ̂= u[7]
    # P = reshape(u[28+1:28+4],2,2);    
    # P = (P+P')/2
    P = u[8]
    Ψ = u[9]

    ϕ̂ = -element(Vush,1,delta_ests[4])

    du[4] = dot(ϕ̂,θ̂) -V  -element(V,afn,delta_ests[1]) -element(Vs,asp,delta_ests[2]) +
                -element(Vs,asn_with_inact,delta_ests[3]) + Iapp

    du[5] = (1/tau_s) * (V - Vs)
    du[6] = (1/tau_us) * (V - Vus)  + γ*(1+Ψ'*P*Ψ)*(Vus-Vush)
    
    du[7]= γ*P*Ψ*(Vus-Vush); # dθ̂ 
    du[9] = -γ*Ψ + ϕ̂;  # dΨ
    du[8] = α*P - ((P*Ψ)*(P*Ψ)');
    # dP = (dP+dP')/2;
    # du[8] = dP[:]
end