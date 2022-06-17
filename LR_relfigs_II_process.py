## Generate the raster plots for the reliability experiment
# on Luka's bursting circuit, using data generated in 'LR_relfigs_burst_gen.jl'.

import matplotlib.pyplot as plt
import numpy as np
import h5py

fname = "sec4_LR_II.jld"

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

# Neuron 1 events
RefEvents1 = convert_to_timings(t,Ref[:,0])
PrelearnEvents1     = []
MisEvents1     = [] # Not used.
LearnedEvents1 = []
for i in range(num_trials):
    PrelearnEvents1.append(convert_to_timings(t,Mis[i,:,0]))
    MisEvents1.append(convert_to_timings(t,Mis[i,:,0]))
    LearnedEvents1.append(convert_to_timings(t,Learned[i,:,0]))

PrelearnEvents1.append(RefEvents1)

# Neuron 2 events
RefEvents2 = convert_to_timings(t,Ref[:,1])
PrelearnEvents2     = []
MisEvents2     = [] # Not used.
LearnedEvents2 = []
for i in range(num_trials):
    PrelearnEvents2.append(convert_to_timings(t,Mis[i,:,1]))
    MisEvents2.append(convert_to_timings(t,Mis[i,:,1]))
    LearnedEvents2.append(convert_to_timings(t,Learned[i,:,1]))

PrelearnEvents2.append(RefEvents2)

# Colouring of plots
from itertools import repeat
mis_color = 'b'
color_list = []
color_list.extend(repeat(mis_color,num_trials))
color_list.append('r') # Ref colour

# plt.eventplot(MisEvents)
fig, ax = plt.subplots(1,1)
l1 = ax.eventplot(PrelearnEvents1,colors=color_list)
l2 = ax.eventplot(LearnedEvents1, colors='g', linelengths=0.7)

fig2, ax2 = plt.subplots(1,1)
l1 = ax2.eventplot(PrelearnEvents2,colors=color_list)
l2 = ax2.eventplot(LearnedEvents2, colors='g', linelengths=0.7)

plt.figure(); plt.plot(t,Mis[0,:,0],t,Mis[0,:,1])
plt.show()
