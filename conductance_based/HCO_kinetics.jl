const ENa = 50.
const EK = -80.
const ECa = 120.
const EL = -49.
const ESyn = -80.

σ(v,r,k) = 1/(1+exp(-(v-r)/k))
τ(v,c1,c2,r,k) = c1 + c2*σ(v,r,k)
τ2(v,c,r,k,c1,c2,r2,k2) = c*σ(v,r,k)*τ(v,c1,c2,r2,k2)

### Na-current
# m activation
rNa_m = -35.5;
kNa_m = 5.29;
σNa_m(V) = σ(V,rNa_m,kNa_m)
# m time constant
c1Na_m = 1.32;
c2Na_m = -1.26;
rτNa_m = -120;
kτNa_m = 25;
τNa_m(V) = τ(V,c1Na_m,c2Na_m,rτNa_m,kτNa_m)
# h activation
rNa_h = -48.9;
kNa_h = -5.18;
σNa_h(V) = σ(V,rNa_h,kNa_h)
# h time constant
cτNa_h = 0.67;
rτNa_h = -62.9;
kτNa_h = 10;
c1Na_h = 1.5;
c2Na_h = 1;
rτ2Na_h = -34.9; 
kτ2Na_h = -3.6;
τNa_h(V) = τ2(V,cτNa_h,rτNa_h,kτNa_h,c1Na_h,c2Na_h,rτ2Na_h,kτ2Na_h)

### K-current
# mK activation
rK_m = -12.3;
kK_m = 11.8;
σK_m(V) = σ(V,rK_m,kK_m)
# mK time constant
c1K_m = 7.2;
c2K_m = -6.4;
rτK_m = -28.3;
kτK_m = 19.2;
τK_m(V) = τ(V,c1K_m,c2K_m,rτK_m,kτK_m)

### Ca-current
# mCa activation
rCa_m = -67.1;          
kCa_m = 7.2;
σCa_m(V)= σ(V,rCa_m,kCa_m)
# mCa time constant
c1Ca_m = 43.4;
c2Ca_m = -42.6;
rτCa_m = -68.1;
kτCa_m = 20.5;
τCa_m(V) = τ(V,c1Ca_m,c2Ca_m,rτCa_m,kτCa_m)
# hCa activation
rCa_h = -82.1;
kCa_h = -5.5;
σCa_h(V) = σ(V,rCa_h,kCa_h)
# hCa time constant
c1Ca_h = 140;
c2Ca_h = -100;
rτCa_h = -55;
kτCa_h = 16.9;
τCa_h(V) = τ(V,c1Ca_h,c2Ca_h,rτCa_h,kτCa_h)

### Synaptic current
rSyn = -45;
kSyn = 2;
σSyn(V) = σ(V,rSyn,kSyn)