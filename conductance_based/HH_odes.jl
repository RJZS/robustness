# Return a value uniformly sampled from x +/- 100*err %
function x_sample(x, err)
    x*(1-err) + 2*err*x*rand()
end

# Sodium activation
function gating_m(v, r=-40.)
    k = 9.;             #15 in izhikevich
    Vmax = -38.;
    std = 30.;
    Camp = 0.46;
    Cbase = 0.04;
    τ = Cbase + Camp*exp(-(v-Vmax).^2/std^2);
    σ = 1 ./ (1+exp(-(v-r)/k));
    return τ, σ
end 

# Sodium inactivation
function gating_h(v, r=-62.)
    k = -7.;
    Vmax = -67.;
    std = 20.;
    Camp = 7.4;
    Cbase = 1.2;
    τ = Cbase + Camp*exp(-(v-Vmax).^2/std^2);
    σ = 1 ./ (1+exp(-(v-r)/k));
    return τ, σ
end

# Potassium activation
function gating_n(v, r=-53.)
    k = 15.;
    Vmax = -79.;
    std = 50.;
    Camp = 4.7;
    Cbase = 1.1;
    τ = Cbase + Camp*exp(-(v-Vmax).^2/std^2);
    σ = 1 ./ (1+exp(-(v-r)/k));
    return τ, σ
end

# Dynamics of synaptic filter
function dsyn(v, s)
    ds = 0.53*(1/(1+exp(-(v-2)/5)))*(1-s)-0.18*s
end

function HH_ode!(dz,z,p,t)
    Iapp =          p[1]
    c =             p[2]
    (gNa,gK,gL) =   p[3]
    (ENa,EK,EL) =   p[4]
    half_acts =     p[5]

    v = z[1]
    m = z[2]
    h = z[3]
    n = z[4]

    (τm,σm) = gating_m(v,half_acts[1]);
    (τh,σh) = gating_h(v,half_acts[2]);
    (τn,σn) = gating_n(v,half_acts[3]);

    g = [gNa; gK; gL];
    phi = [-m^3*h*(v-ENa);-n^4*(v-EK);-(v-EL)];

    dz[1] = 1/c * (dot(phi,g) + Iapp(t));
    dz[2] = 1/τm*(-m + σm);
    dz[3] = 1/τh*(-h + σh);
    dz[4] = 1/τn*(-n + σn);
end

function HH_ode_noise!(dz,z,p,t)
    dz = [2 0 0 0];
end
    

function HH_observer_orig!(dz,z,p,t)
    Iapp =          p[1]
    c =             p[2]
    (gNa,gK,gL) =   p[3]
    (ENa,EK,EL) =   p[4]
    (α,γ) =         p[5]
    (rm, rh, rn) =  p[6]

    # True system
    v = z[1]
    m = z[2]
    h = z[3]
    n = z[4]

    (τm,σm) = gating_m(v);
    (τh,σh) = gating_h(v);
    (τn,σn) = gating_n(v);

    θ = 1/c*[gNa gK gL gNa*ENa gK*EK gL*EL 1]
    ϕ = [-m^3*h*v ...
         -n^4*v ... 
         -v ...
         m^3*h ...
         n^4 ...
         1 ...
         Iapp(t)];

    dv = dot(ϕ,θ)
    dm = 1/τm*(-m + σm);
    dh = 1/τh*(-h + σh);
    dn = 1/τn*(-n + σn);

    # Adaptive observer
    v̂ = z[5]
    m̂ = z[6]
    ĥ = z[7]
    n̂ = z[8]
    θ̂ = z[9:15]
    P = reshape(z[15+1:15+49],7,7);    
    P = (P+P')/2
    Ψ = z[15+49+1:15+49+7]

    (τm̂,σm̂) = gating_m(v, rm);
    (τĥ,σĥ) = gating_h(v, rh);
    (τn̂,σn̂) = gating_n(v, rn);

    ϕ̂ = [-m̂^3*ĥ*v ...
         -n̂^4*v ... 
         -v ...
         m̂^3*ĥ ...
         n̂^4 ...
         1 ...
         Iapp(t)];

    dv̂ = dot(ϕ̂,θ̂) + γ*(1+Ψ'*P*Ψ)*(v-v̂)
    dm̂ = 1/τm̂*(-m̂ + σm̂);
    dĥ = 1/τĥ*(-ĥ + σĥ);
    dn̂ = 1/τn̂*(-n̂ + σn̂);

    dθ̂ = γ*P*Ψ*(v-v̂);
    dΨ = -γ*Ψ + ϕ̂; 
    dP = α*P - ((P*Ψ)*(P*Ψ)');
    dP = (dP+dP')/2;

    dz[:] = [dv;dm;dh;dn;dv̂;dm̂;dĥ;dn̂;dθ̂;dP[:];dΨ]';
end

function HH_s_observer!(dz,z,p,t)
    Iapp =          p[1]
    c =             p[2]
    (gNa,gK,gL) =   p[3]
    (ENa,EK,EL) =   p[4]
    (α,γ) =         p[5]
    (rm, rh, rn) =  p[6]

    # True system
    v = z[1]
    m = z[2]
    h = z[3]
    n = z[4]
    s = z[72]

    (τm,σm) = gating_m(v);
    (τh,σh) = gating_h(v);
    (τn,σn) = gating_n(v);

    θ = 1/c*[gNa gK gL gNa*ENa gK*EK gL*EL 1]
    ϕ = [-m^3*h*v ...
         -n^4*v ... 
         -v ...
         m^3*h ...
         n^4 ...
         1 ...
         Iapp(t)];

    dv = dot(ϕ,θ) - s
    dm = 1/τm*(-m + σm);
    dh = 1/τh*(-h + σh);
    dn = 1/τn*(-n + σn);
    ds = dsyn(v, s)

    # Adaptive observer
    v̂ = z[5]
    m̂ = z[6]
    ĥ = z[7]
    n̂ = z[8]
    θ̂ = z[9:15]
    P = reshape(z[15+1:15+49],7,7);    
    P = (P+P')/2
    Ψ = z[15+49+1:15+49+7]
    s_hat = z[73]

    (τm̂,σm̂) = gating_m(v̂, rm);
    (τĥ,σĥ) = gating_h(v̂, rh);
    (τn̂,σn̂) = gating_n(v̂, rn);

    ϕ̂ = [-m̂^3*ĥ*v̂ ...
         -n̂^4*v̂ ... 
         -v̂ ...
         m̂^3*ĥ ...
         n̂^4 ...
         1 ...
         Iapp(t)];

    dv̂ = dot(ϕ̂,θ̂) - s_hat # Shouldn't this be s?
    dm̂ = 1/τm̂*(-m̂ + σm̂);
    dĥ = 1/τĥ*(-ĥ + σĥ);
    dn̂ = 1/τn̂*(-n̂ + σn̂);
    ds_hat = dsyn(v̂, s) + γ*(1+Ψ'*P*Ψ)*(s-s_hat)

    dθ̂ = γ*P*Ψ*(v-v̂);
    dΨ = -γ*Ψ + ϕ̂; 
    dP = α*P - ((P*Ψ)*(P*Ψ)');
    dP = (dP+dP')/2;

    dz[:] = [dv;dm;dh;dn;dv̂;dm̂;dĥ;dn̂;dθ̂;dP[:];dΨ;ds;ds_hat]';
end

function HH_observer!(dz,z,p,t)
    Iapp =          p[1]
    c =             p[2]
    (gNa,gK,gL) =   p[3]
    (ENa,EK,EL) =   p[4]
    (α,γ) =         p[5]
    (rm, rh, rn) =  p[6]
    sensor_noise =  p[7]

    # True system
    v = z[1]
    m = z[2]
    h = z[3]
    n = z[4]

    (τm,σm) = gating_m(v);
    (τh,σh) = gating_h(v);
    (τn,σn) = gating_n(v);

    θ = 1/c*[gNa gK gL gNa*ENa gK*EK gL*EL 1]
    ϕ = [-m^3*h*v ...
         -n^4*v ... 
         -v ...
         m^3*h ...
         n^4 ...
         1 ...
         Iapp(t)];

    dv = dot(ϕ,θ)
    dm = 1/τm*(-m + σm);
    dh = 1/τh*(-h + σh);
    dn = 1/τn*(-n + σn);

    # Adaptive observer
    v̂ = z[5]
    m̂ = z[6]
    ĥ = z[7]
    n̂ = z[8]
    θ̂ = z[9:11]
    P = reshape(z[11+1:11+9],3,3);    
    P = (P+P')/2
    Ψ = z[11+9+1:11+9+3]

    # Add measurement noise
    # nv = v + sensor_noise(t)
    nv = v

    (τm̂,σm̂) = gating_m(nv, rm);
    (τĥ,σĥ) = gating_h(nv, rh);
    (τn̂,σn̂) = gating_n(nv, rn);

    ϕ̂ =   (1/c)*[-m̂^3*ĥ*(nv-ENa) ...
                -n̂^4*(nv-EK) ... 
                -(nv-EL)];

    dv̂ = dot(ϕ̂,θ̂) + Iapp(t) + γ*(1+Ψ'*P*Ψ)*(nv-v̂)
    dm̂ = 1/τm̂*(-m̂ + σm̂);
    dĥ = 1/τĥ*(-ĥ + σĥ);
    dn̂ = 1/τn̂*(-n̂ + σn̂);

    dθ̂ = γ*P*Ψ*(nv-v̂);
    dΨ = -γ*Ψ + ϕ̂; 
    dP = α*P - ((P*Ψ)*(P*Ψ)');
    dP = (dP+dP')/2;

    dz[:] = [dv;dm;dh;dn;dv̂;dm̂;dĥ;dn̂;dθ̂;dP[:];dΨ]';
end