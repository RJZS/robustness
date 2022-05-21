# The model introduced in "Neuromodulation of Neuromorphic Circuits"
# by Luka Ribar.

using DifferentialEquations

function sigmoid(x, k=1)
    1 ./ (1 .+ exp(-k * x))
end

function element(V, a, delta)
    a * tanh(V - delta)
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
end

# Original, v observer.
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
    θ̂= u[7]
    # P = reshape(u[28+1:28+4],2,2);    
    # P = (P+P')/2
    P = u[8]
    Ψ = u[9]

    ϕ̂ = -element(Vush,1,delta_ests[4])

    du[4] = dot(ϕ̂,θ̂) -V  -element(V,afn,delta_ests[1]) -element(Vsh,asp,delta_ests[2]) +
                -element(Vsh,asn,delta_ests[3]) + Iapp +
                γ*(1+Ψ'*P*Ψ)*(V-Vh)

    du[5] = (1/tau_s) * (V - Vsh)
    du[6] = (1/tau_us) * (V - Vush)
    
    du[7]= γ*P*Ψ*(V-Vh); # dθ̂ 
    du[9] = -γ*Ψ + ϕ̂;  # dΨ
    du[8] = α*P - ((P*Ψ)*(P*Ψ)');
    # dP = (dP+dP')/2;
    # du[8] = dP[:]
end