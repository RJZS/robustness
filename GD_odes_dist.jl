using DifferentialEquations, Plots, Plots.PlotMeasures, LaTeXStrings

function CBM_v_dist_observer!(du,u,p,t)
    # Stimulations parameters
    Iapp=p[1] # Amplitude of constant applied current
    I1=p[2] # Amplitude of first step input
    I2=p[3] # Amplitude of second step input
    ti1=p[4] # Starting time of first step input
    tf1=p[5] # Ending time of first step input
    ti2=p[6] # Starting time of second step input
    tf2=p[7] # Ending time of second step input
    
    # Maximal conductances
    gNa=p[8] # Sodium current maximal conductance
    gKd=p[9]  # Delayed-rectifier potassium current maximal conductance
    gAf=p[10] # Fast A-type potassium current maximal conductance
    gAs=p[11] # Slow A-type potassium current maximal conductance
    gKCa=p[12] # Calcium-activated potassium current maximal conductance
    gCaL=p[13] # L-type calcium current maximal conductance
    gCaT=p[14] # T-type calcium current maximal conductance
    gH=p[15] # H-current maximal conductance
    gl=p[16] # Leak current maximal conductance
    
    half_acts = p[17]
    ha_ts = p[18] # Half_act taus (for observer model error)
    α1 = p[19] # Alpha during first prat of simulation

    # Variables
    V=u[1] # Membrane potential
    mNa=u[2] # Sodium current activation
    hNa=u[3] # Sodium current inactivation
    mKd=u[4] # Delayed-rectifier potassium current activation
    mAf=u[5] # Fast A-type potassium current activation
    hAf=u[6] # Fast A-type potassium current inactivation
    mAs=u[7] # Slow A-type potassium current activation
    hAs=u[8] # Slow A-type potassium current inactivation
    mCaL=u[9] # L-type calcium current activation
    mCaT=u[10] # T-type calcium current activation
    hCaT=u[11] # T-type calcium current inactivation
    mH=u[12] # H current activation
    Ca=u[13] # Intracellular calcium concentration
    
    θ = [gCaL gCaT]
    ϕ = 1/C*[-mCaL*(V-VCa) ...
            -mCaT*hCaT*(V-VCa)];
        
                # Sodium current
    b = (1/C) * (-gNa*mNa*hNa*(V-VNa) +
                # Potassium Currents
                -gKd*mKd*(V-VK) -gAf*mAf*hAf*(V-VK) -gAs*mAs*hAs*(V-VK) +
                -gKCa*mKCainf(Ca)*(V-VK) +
                # Cation current
                -gH*mH*(V-VH) +
                # Passive currents
                -gl*(V-Vl) +
                # Stimulation currents
                +Iapp(t) + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2))
    du[1] = dot(ϕ,θ) + b

    # Internal dynamics
    du[2] = (1/tau_mNa(V)) * (mNainf(V) - mNa)
    du[3] = (1/tau_hNa(V)) * (hNainf(V) - hNa)
    du[4] = (1/tau_mKd(V)) * (mKdinf(V) - mKd)
    du[5] = (1/tau_mAf(V)) * (mAfinf(V) - mAf)
    du[6] = (1/tau_hAf(V)) * (hAfinf(V) - hAf)
    du[7] = (1/tau_mAs(V)) * (mAsinf(V) - mAs)
    du[8] = (1/tau_hAs(V)) * (hAsinf(V) - hAs)
    du[9] = (1/tau_mCaL(V)) * (mCaLinf(V) - mCaL)
    du[10] = (1/tau_mCaT(V)) * (mCaTinf(V) - mCaT)
    du[11] = (1/tau_hCaT(V)) * (hCaTinf(V) - hCaT)
    du[12] = (1/tau_mH(V)) * (mHinf(V) - mH)
    du[13] = (1/tau_Ca) * ((-αCa*gCaL*mCaL*(V-VCa))+(-β*gCaT*mCaT*hCaT*(V-VCa)) - Ca) 

    # Adaptive observer
    Vh=u[14] # Membrane potential
    mNah=u[15] # Sodium current activation
    hNah=u[16] # Sodium current inactivation
    mKdh=u[17] # Delayed-rectifier potassium current activation
    mAfh=u[18] # Fast A-type potassium current activation
    hAfh=u[19] # Fast A-type potassium current inactivation
    mAsh=u[20] # Slow A-type potassium current activation
    hAsh=u[21] # Slow A-type potassium current inactivation
    mCaLh=u[22] # L-type calcium current activation
    mCaTh=u[23] # T-type calcium current activation
    hCaTh=u[24] # T-type calcium current inactivation
    mHh=u[25] # H current activation
    Cah=u[26] # Intracellular calcium concentration
    θ̂= u[27:31]
    P = u[31+1:31+5]
    Ψ = u[31+5+1:31+5+5]

    if t < 5000
        α = α1
    else
        α = 0
    end

    ϕ̂ = 1/C*[-mNah*hNah*(V-VNa);
            -mKdh*(V-VK);
            -mKCainf(Cah,half_acts[12])*(V-VK);
            -mCaLh * (V-VCa);
            -mCaTh * hCaTh * (V-VCa)]

    bh = (1/C) * (
    # Potassium Currents
    -gAf*mAfh*hAfh*(V-VK) -gAs*mAsh*hAsh*(V-VK) +
    # Cation current
    -gH*mHh*(V-VH) +
    # Passive currents
    -gl*(V-Vl) +
    # Stimulation currents
    +Iapp(t) + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2))

    # NOTE: No transpose in Psi as everything is 1D.

    # dV^
    du[14] = dot(ϕ̂',θ̂) + bh + γ0 + sum((γs .* Ψ .* P .* Ψ))*(V-Vh)

    # Internal dynamics
    du[15] = (1/tau_mNa(V,ha_ts[1])) * (mNainf(V,half_acts[1]) - mNah)
    du[16] = (1/tau_hNa(V,ha_ts[2])) * (hNainf(V,half_acts[2]) - hNah)
    du[17] = (1/tau_mKd(V,ha_ts[3])) * (mKdinf(V,half_acts[3]) - mKdh)
    du[18] = (1/tau_mAf(V,ha_ts[4])) * (mAfinf(V,half_acts[4]) - mAfh)
    du[19] = (1/tau_hAf(V,ha_ts[5])) * (hAfinf(V,half_acts[5]) - hAfh)
    du[20] = (1/tau_mAs(V,ha_ts[6])) * (mAsinf(V,half_acts[6]) - mAsh)
    du[21] = (1/tau_hAs(V,ha_ts[7])) * (hAsinf(V,half_acts[7]) - hAsh)
    du[22] = (1/tau_mCaL(V,ha_ts[8])) * (mCaLinf(V,half_acts[8]) - mCaLh)
    du[23] = (1/tau_mCaT(V,ha_ts[9])) * (mCaTinf(V,half_acts[9]) - mCaTh)
    du[24] = (1/tau_hCaT(V,ha_ts[10])) * (hCaTinf(V,half_acts[10]) - hCaTh)
    du[25] = (1/tau_mH(V,ha_ts[11])) * (mHinf(V,half_acts[11]) - mHh)
    du[26] = (1/tau_Ca) * ((-(αCa)*θ̂[4]*mCaLh*(V-VCa))+(-(β)*θ̂[5]*mCaTh*hCaTh*(V-VCa)) - Cah) 

    # Can implement a different alpha for each theta.
    du[27:31]= γs .* P .* Ψ * (V-Vh); # dθ̂ 
    du[31+5+1:31+5+5] = -γs .* Ψ .+ ϕ̂;  # dΨ
    du[31+1:31+5] = α1*P .- α1*P.*Ψ.*Ψ.*P # dP
end