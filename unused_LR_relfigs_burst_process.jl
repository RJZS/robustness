## Generate the raster plots for the reliability experiment
# on Luka's bursting circuit, using data generated in 'LR_relfigs_burst_gen.jl'.

using Plots, JLD

fname = "sec4_LR_burst.jld"

noise_values     = load(fname)["noise_values"]
delta_est_values = load(fname)["delta_est_values"]
thetalearned     = load(fname)["thetalearned"]

ts      = load(fname)["ts"]
Ref     = load(fname)["Ref"]
Mis     = load(fname)["Mis"]
Learned = load(fname)["Learned"]

function convert_to_timings(t, y, thresh=0)
    event_idxs = findall(x -> x > thresh, y)
    event_times = t[event_idxs]
end