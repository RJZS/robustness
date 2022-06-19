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

## Stimulation function
heaviside(t)=(1+sign(t))/2 # Unit step function
pulse(t,ti,tf)=heaviside(t-ti)-heaviside(t-tf) # Pulse function

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

    I3 = p[18]
    ti3 = p[19]
    tf3 = p[20]

    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    # ODEs
    du[1] = -V  -element(V,afn,dfn) -element(Vs,asp,dsp) +
                -element(Vs,asn,dsn) -element(Vus,ausp,dusp) +
                Iapp + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2) +
                I3*pulse(t,ti3,tf3)
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)
end

# Deprecated (no inact)
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
    I5 = p[33]
    ti5 = p[34]
    tf5 = p[35]

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
                Iapp2 + I3*pulse(t,ti3,tf3) + I4*pulse(t,ti4,tf4) + I5*pulse(t,ti5,tf5)
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

function LR_ODE_rel_II!(du,u,p,t)
    # Gains
    afn = p[1]
    asp = p[2]
    asn = p[3]
    ausp = p[4]

    # Gains of second neuron
    afn2 = p[5]
    asp2 = p[6]
    asn2 = p[7]
    ausp2 = p[8]
    
    # Time constants
    tau_s = p[9]
    tau_us = p[10]

    # Inputs
    noise = p[11]
    Iapp2 = p[12]

    # Synaptic parameters
    asyn21 = p[13]
    asyn12 = p[14]

    deltas = p[15] # [dfn1 dsp1 dsn1 dusp1 delta_h1 dfn2 ... dsyn21 dsyn12]
    beta = p[16]


    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    V2 = u[4]
    Vs2 = u[5]
    Vus2 = u[6]

    # ODEs
    asn_with_inact = asn * sigmoid(-(Vus-deltas[5]),beta)
    du[1] = -V  -element(V,afn,deltas[1]) -element(Vs,asp,deltas[2]) +
                -element(Vs,asn_with_inact,deltas[3]) -element(Vus,ausp,deltas[4]) +
                synapse(Vs2, asyn12, deltas[12], beta) +
                noise[round(Int, t/dt)+1]
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)

    asn2_with_inact = asn2 * sigmoid(-(Vus2-deltas[10]),beta)
    du[4] = -V2  -element(V2,afn2,deltas[6]) -element(Vs2,asp2,deltas[7]) +
                -element(Vs2,asn2_with_inact,deltas[8]) -element(Vus2,ausp2,deltas[9]) +
                synapse(Vs, asyn21, deltas[11], beta) +
                Iapp2
    du[5] = (1/tau_s)  * (V2 - Vs2)
    du[6] = (1/tau_us) * (V2 - Vus2)
end

# Original, v observer.
function LR_observer_noinact_nondiag!(du,u,p,t)
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
    
    α1   = p[13]
    γ   = p[14]
    
    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    # ODEs
    du[1] = -V  -element(V,afn,dfn) -element(Vs,asp,dsp) +
                -element(Vs,asn,dsn) -element(Vus,ausp,dusp) +
                Iapp(t)
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)

    # Adaptive Observer
    Vh = u[4]
    Vsh = u[5]
    Vush = u[6]
    θ̂= u[7:10]
    P = reshape(u[10+1:10+16],4,4);    
    P = (P+P')/2
    # P = u[11:14]
    Ψ = u[27:30]

    t > 30000 ? α = 0 : α = α1

    ϕ̂ = [-element(V,1,delta_ests[1]) ...
        -element(Vsh,1,delta_ests[2]) ...
        -element(Vsh,1,delta_ests[3]) ...
        -element(Vush,1,delta_ests[4])]

    du[4] = dot(ϕ̂,θ̂) -V  + Iapp(t) + γ*(1+Ψ'*P*Ψ)*(V-Vh)

    du[5] = (1/tau_s) * (V - Vsh)
    du[6] = (1/tau_us) * (V - Vush)
    
    du[7:10]= γ*P*Ψ*(V-Vh); # dθ̂ 
    du[27:30] = -γ*Ψ + ϕ̂;  # dΨ
    dP = α*P - γ*((P*Ψ)*(P*Ψ)');
    dP = (dP+dP')/2;
    du[10+1:10+16] = dP[:]
end

# Diagonalised version of v observer.
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
    
    α1   = p[13]
    γ   = p[14]
    
    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    # ODEs
    du[1] = -V  -element(V,afn,dfn) -element(Vs,asp,dsp) +
                -element(Vs,asn,dsn) -element(Vus,ausp,dusp) +
                Iapp(t)
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

    t > 40000 ? α = 0 : α = α1

    ϕ̂ = [-element(V,1,delta_ests[1]) ...
        -element(Vsh,1,delta_ests[2]) ...
        -element(Vsh,1,delta_ests[3]) ...
        -element(Vush,1,delta_ests[4])]

    du[4] = dot(ϕ̂,θ̂) -V  + Iapp(t) + γ*(1+sum(P.*Ψ.^2))*(V-Vh) # γ*(1+Ψ'*P*Ψ)*(V-Vh)

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

function LR_observer_II!(du,u,p,t)
    # Gains
    afn = p[1]
    asp = p[2]
    asn = p[3]
    ausp = p[4]

    # Gains of second neuron
    afn2 = p[5]
    asp2 = p[6]
    asn2 = p[7]
    ausp2 = p[8]
    
    # Time constants
    tau_s = p[9]
    tau_us = p[10]

    # Inputs
    Iapp = p[11]
    Iapp2 = p[12]

    # Synaptic parameters
    asyn21 = p[13]
    asyn12 = p[14]

    deltas = p[15] # [dfn1 dsp1 dsn1 dusp1 delta_h1 dfn2 ... dsyn21 dsyn12]
    delta_ests = p[16]
    beta = p[17]
    
    α1   = p[18]
    γ   = p[19]

    # Variables
    V = u[1]
    Vs = u[2]
    Vus = u[3]

    V2 = u[4]
    Vs2 = u[5]
    Vus2 = u[6]

    # ODEs
    asn_with_inact = asn * sigmoid(-(Vus-deltas[5]),beta)
    du[1] = -V  -element(V,afn,deltas[1]) -element(Vs,asp,deltas[2]) +
                -element(Vs,asn_with_inact,deltas[3]) -element(Vus,ausp,deltas[4]) +
                synapse(Vs2, asyn12, deltas[12], beta) +
                Iapp(t)
    du[2] = (1/tau_s)  * (V - Vs)
    du[3] = (1/tau_us) * (V - Vus)

    asn2_with_inact = asn2 * sigmoid(-(Vus2-deltas[10]),beta)
    du[4] = -V2  -element(V2,afn2,deltas[6]) -element(Vs2,asp2,deltas[7]) +
                -element(Vs2,asn2_with_inact,deltas[8]) -element(Vus2,ausp2,deltas[9]) +
                synapse(Vs, asyn21, deltas[11], beta) +
                Iapp2
    du[5] = (1/tau_s)  * (V2 - Vs2)
    du[6] = (1/tau_us) * (V2 - Vus2)

    # Adaptive Observer
    Vh = u[7]
    Vsh = u[8]
    Vush = u[9]

    V2h = u[10]
    Vs2h = u[11]
    Vus2h = u[12]

    θ̂= u[13:22]
    # P = reshape(u[28+1:28+4],2,2);    
    # P = (P+P')/2

    P = u[23:27]
    P2 = u[28:32]
    Ψ = u[33:37]
    Ψ2 = u[38:42]

    t > 30000 ? α = 0 : α = α1

    ϕ̂ = [-element(V,1,delta_ests[1]) ...
        -element(Vsh,1,delta_ests[2]) ...
        -element(Vsh,1,delta_ests[3])*sigmoid(-(Vush-delta_ests[5]),beta) ...
        -element(Vush,1,delta_ests[4]) ...
        synapse(Vs2h, 1, delta_ests[12], beta)]

    du[7] = dot(ϕ̂,θ̂[1:5]) -V  + Iapp(t) + 
            γ*(1+sum(P.*Ψ.^2))*(V-Vh) # γ*(1+Ψ'*P*Ψ)*(V-Vh)

    du[8] = (1/tau_s) * (V - Vsh)
    du[9] = (1/tau_us) * (V - Vush)

    ϕ̂2 = [-element(V2,1,delta_ests[6]) ...
        -element(Vs2h,1,delta_ests[7]) ...
        -element(Vs2h,1,delta_ests[8])* sigmoid(-(Vus2h-delta_ests[10]),beta) ...
        -element(Vus2h,1,delta_ests[9]) ...
        synapse(Vsh, 1, delta_ests[11], beta)]

    du[10] = dot(ϕ̂2,θ̂[6:10]) -V2  + Iapp2 + 
            γ*(1+sum(P2.*Ψ2.^2))*(V2-V2h) # γ*(1+Ψ'*P*Ψ)*(V-Vh)

    du[11] = (1/tau_s) * (V2 - Vs2h)
    du[12] = (1/tau_us) * (V2 - Vus2h)
    
    du[13:17]= γ*P.*Ψ*(V-Vh); # dθ̂  (neuron 1)
    du[18:22]= γ*P2.*Ψ2*(V2-V2h); # dθ̂  (neuron 2)

    du[33:37] = -γ*Ψ + ϕ̂;  # dΨ1
    du[38:42] = -γ*Ψ2 + ϕ̂2;  # dΨ2

    du[23:27] = α*P - α*P.^2 .* Ψ.^2; # ((P*Ψ)*(P*Ψ)'); Neuron 1
    du[28:32] = α*P2 - α*P2.^2 .* Ψ2.^2; # ((P*Ψ)*(P*Ψ)'); Neuron 2
    # dP = (dP+dP')/2;
    # du[8] = dP[:]
end