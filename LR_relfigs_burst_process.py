## Generate the raster plots for the reliability experiment
# on Luka's bursting circuit, using data generated in 'LR_relfigs_burst_gen.jl'.

import matplotlib.pyplot as plt
import numpy as np
import h5py

fname = "sec4_LR_burst.jld"

f = h5py.File(fname, "r")

noise     = f.get("noise")
delta_est_values = f.get("delta_est_values")
thetalearned     = f.get("thetalearned")

t      = f.get("t")
Ref     = f.get("Ref")
Mis     = f.get("Mis")
Learned = f.get("Learned")

t=np.array(t); Ref=np.array(Ref); Mis=np.array(Mis); Learned=np.array(Learned)
num_trials = Mis.shape[0]
def convert_to_timings(t, y, thresh=0):
    event_idxs = np.where(y>thresh)
    event_times = t[event_idxs]
    # Use np.diffs to eliminate all but the first spike in each burst.
    return event_times

RefEvents = convert_to_timings(t,Ref)
PrelearnEvents     = []
MisEvents     = [] # Not used.
LearnedEvents = []
for i in range(num_trials):
    PrelearnEvents.append(convert_to_timings(t,Mis[i,:]))
    MisEvents.append(convert_to_timings(t,Mis[i,:]))
    LearnedEvents.append(convert_to_timings(t,Learned[i,:]))

PrelearnEvents.append(RefEvents)

# Colouring of plots
from itertools import repeat
mis_color = 'b'
color_list = []
color_list.extend(repeat(mis_color,num_trials))
color_list.append('r') # Ref colour

# plt.eventplot(MisEvents)
fig, ax = plt.subplots(1,1)
l1 = ax.eventplot(PrelearnEvents,colors=color_list)
l2 = ax.eventplot(LearnedEvents, colors='g', linelengths=0.7)
plt.show()