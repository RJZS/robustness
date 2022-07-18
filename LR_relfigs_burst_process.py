## Generate the raster plots for the reliability experiment
# on Luka's bursting circuit, using data generated in 'LR_relfigs_burst_gen.jl'.

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
noise=np.array(noise)
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

dpi = 200
# plt.eventplot(MisEvents)
fig = plt.figure(constrained_layout=True,dpi=dpi)
gs = GridSpec(3, 2, figure=fig, height_ratios=[1,0.4,1])
ax1 = fig.add_subplot(gs[0, :])
# fig, ax = plt.subplots(1,1)
l1 = ax1.eventplot(PrelearnEvents,colors=color_list)
l2 = ax1.eventplot(LearnedEvents, colors='tab:orange', linelengths=0.7)
ax1.set_yticks([1,2,3,4,5,6,7,8])
ax1.axis(xmin=t[0]-1000, xmax=t[-1]+1000,ymin=0.2,ymax=9.8)
ax1.set_xlabel("t")
ax1.set_ylabel("Trial")
# plt.xlabel("t")
# plt.ylabel("Trial")
# plt.savefig("sec5_LR_bursting_raster.png")

# Current plot
axI = fig.add_subplot(gs[1,:])
axI.plot(t[:-1], noise)
axI.set_ylabel("I")
axI.set_xlabel("t")

blend = 0.5 # For alpha parameter of Line2D

# Time-series plot
num_samps = 8 # Number of trials to plot.
# fig, axs = plt.subplots(1, 2, sharey=True)
ax2 = fig.add_subplot(gs[2,0])
ax3 = fig.add_subplot(gs[2,1])
ax2.plot(t, Ref, color='tab:red')
for i in range(num_samps):
    ax2.plot(t, Mis[i,:], color='tab:blue', alpha=blend)
ax3.plot(t, Ref, color='tab:red')
for i in range(num_samps):
    ax3.plot(t, Learned[i,:], color='tab:orange', alpha=blend)
ax2.set_xlabel("t")
ax3.set_xlabel("t")
ax2.set_ylabel("V")
# plt.savefig("sec5_LR_bursting_timeseries.png")

plt.savefig("sec5_LR_bursting.png")

# Raster for step input
figStep = plt.figure(constrained_layout=True,dpi=dpi)
gsStep = GridSpec(3, 2, figure=figStep, height_ratios=[1,0.4,1])
ax1 = figStep.add_subplot(gsStep[0, :])
l1 = ax1.eventplot(PrelearnEventsStep,colors=color_list)
l2 = ax1.eventplot(LearnedEventsStep, colors='tab:orange', linelengths=0.7)
ax1.set_yticks([1,2,3,4,5,6])
ax1.axis(xmin=t[0]-1000, xmax=t[-1]+1000,ymin=0.2,ymax=9.8)
ax1.set_xlabel("t")
ax1.set_ylabel("Trial")
# plt.savefig("sec5_LR_bursting_raster_step.png")

# Current plot
axI = figStep.add_subplot(gsStep[1,:])
Istep = -2.2*np.ones(int(20000/0.1+1))
Istep[0:30000] = -2.6
axI.plot(t[:-1], Istep)
axI.set_ylabel("I")
axI.set_xlabel("t")

# Time-series plot for step input
ax2 = figStep.add_subplot(gsStep[2,0])
ax3 = figStep.add_subplot(gsStep[2,1])
ax2.plot(t, RefStep, color='tab:red')
for i in range(num_samps):
    ax2.plot(t, MisStep[i,:], color='tab:blue', alpha=blend)
ax3.plot(t, RefStep, color='tab:red')
for i in range(num_samps):
    ax3.plot(t, LearnedStep[i,:], color='tab:orange', alpha=blend)
ax2.set_xlabel("t")
ax3.set_xlabel("t")
ax2.set_ylabel("V")
# plt.savefig("sec5_LR_bursting_timeseries_step.png")

plt.savefig("sec5_LR_bursting_step.png")

plt.show()
