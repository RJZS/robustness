## Generate the raster plots for the reliability experiment
# on Luka's bursting circuit, using data generated in 'LR_relfigs_burst_gen.jl'.

import matplotlib.pyplot as plt
import numpy as np
import h5py

fname = "sec4_LR_burst_withstep.jld"

f = h5py.File(fname, "r")

noise     = f.get("noise")
delta_est_values = f.get("delta_est_values")
thetalearned     = f.get("thetalearned")

t      = f.get("t")
Ref     = f.get("Ref")
Mis     = f.get("Mis")
Learned = f.get("Learned")
RefStep     = f.get("RefStep")
MisStep     = f.get("MisStep")
LearnedStep = f.get("LearnedStep")

t=np.array(t); Ref=np.array(Ref); Mis=np.array(Mis); Learned=np.array(Learned)
RefStep=np.array(RefStep); MisStep=np.array(MisStep); LearnedStep=np.array(LearnedStep)
num_trials = Mis.shape[0]

def convert_to_timings(t, y, thresh=0):
    event_idxs = np.where(y>thresh)
    event_times = t[event_idxs]
    # Use np.diffs to eliminate all but the first spike in each burst.
    return event_times

RefEvents = convert_to_timings(t,Ref)
PrelearnEvents     = [[]] # The double brackets are to 1-index the trials, by including a blank one at 0.
MisEvents     = [[]] # Not used.
LearnedEvents = [[]]
for i in range(num_trials):
    PrelearnEvents.append(convert_to_timings(t,Mis[i,:]))
    MisEvents.append(convert_to_timings(t,Mis[i,:]))
    LearnedEvents.append(convert_to_timings(t,Learned[i,:]))
PrelearnEvents.append(RefEvents)

# And the same again for the step
RefEventsStep = convert_to_timings(t,RefStep)
PrelearnEventsStep     = [[]] # The double brackets are to 1-index the trials, by including a blank one at 0.
MisEventsStep     = [[]] # Not used.
LearnedEventsStep = [[]]
for i in range(num_trials):
    PrelearnEventsStep.append(convert_to_timings(t,MisStep[i,:]))
    MisEventsStep.append(convert_to_timings(t,MisStep[i,:]))
    LearnedEventsStep.append(convert_to_timings(t,LearnedStep[i,:]))
PrelearnEventsStep.append(RefEventsStep)

# Colouring of plots
from itertools import repeat
mis_color = 'tab:blue'
color_list = []
color_list.extend(repeat(mis_color,num_trials+1))
color_list.append('tab:red') # Ref colour

# plt.eventplot(MisEvents)
fig, ax = plt.subplots(1,1)
l1 = ax.eventplot(PrelearnEvents,colors=color_list)
l2 = ax.eventplot(LearnedEvents, colors='tab:orange', linelengths=0.7)
ax.set_yticks([1,2,3,4,5,6,7,8])
ax.axis(xmin=t[0], xmax=t[-1],ymin=0.2,ymax=9.8)
plt.xlabel("t")
plt.ylabel("Trial")
plt.savefig("sec5_LR_bursting_raster.png")

# Example plot
eg_sim = 2 # Which simulation to use for the example plot
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(t, Ref, color='tab:red')
axs[0].plot(t, Mis[eg_sim,:], color='tab:blue')
axs[1].plot(t, Ref, color='tab:red')
axs[1].plot(t, Learned[eg_sim,:], color='tab:orange')
axs[0].set_xlabel("t")
axs[1].set_xlabel("t")
axs[0].set_ylabel("V")
plt.savefig("sec5_LR_bursting_example.png")

# Raster for step input
fig, ax = plt.subplots(1,1)
l1 = ax.eventplot(PrelearnEventsStep,colors=color_list)
l2 = ax.eventplot(LearnedEventsStep, colors='tab:orange', linelengths=0.7)
ax.set_yticks([1,2,3,4,5,6])
ax.axis(xmin=t[0], xmax=t[-1],ymin=0.2,ymax=9.8)
plt.xlabel("t")
plt.ylabel("Trial")
plt.savefig("sec5_LR_bursting_raster_step.png")

plt.show()
