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