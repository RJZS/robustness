## Generate the raster plots for the reliability experiment
# on Luka's bursting circuit, using data generated in 'LR_relfigs_burst_gen.jl'.

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import h5py

# fname = "sec4_LR_II_oldpaperversion.jld" # Version previously used in paper.
fname = "sec4_LR_II_noinact_withstep.jld" # From LR_Relfigs_II_noinact_gen.jl

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

t = t / 1000
num_trials = Mis.shape[0]

def convert_to_timings(t, y, thresh=0):
    event_idxs = np.where(y>thresh)
    event_times = t[event_idxs]
    # Use np.diffs to eliminate all but the first spike in each burst.
    return event_times

# Neuron 1 events
RefEvents1 = convert_to_timings(t,Ref[:,0])
PrelearnEvents1     = [[]] # The double brackets are to 1-index the trials, by including a blank one at 0.
MisEvents1     = [[]] # Not used.
LearnedEvents1 = [[]]
for i in range(num_trials):
    PrelearnEvents1.append(convert_to_timings(t,Mis[i,:,0]))
    MisEvents1.append(convert_to_timings(t,Mis[i,:,0]))
    LearnedEvents1.append(convert_to_timings(t,Learned[i,:,0]))
PrelearnEvents1.append(RefEvents1)

# Neuron 2 events
RefEvents2 = convert_to_timings(t,Ref[:,1])
PrelearnEvents2     = [[]]
MisEvents2     = [[]] # Not used.
LearnedEvents2 = [[]]
for i in range(num_trials):
    PrelearnEvents2.append(convert_to_timings(t,Mis[i,:,1]))
    MisEvents2.append(convert_to_timings(t,Mis[i,:,1]))
    LearnedEvents2.append(convert_to_timings(t,Learned[i,:,1]))
PrelearnEvents2.append(RefEvents2)

# And the step
# Neuron 1 step events
RefEvents1Step = convert_to_timings(t,RefStep[:,0])
PrelearnEvents1Step     = [[]] # The double brackets are to 1-index the trials, by including a blank one at 0.
MisEvents1Step     = [[]] # Not used.
LearnedEvents1Step = [[]]
for i in range(num_trials):
    PrelearnEvents1Step.append(convert_to_timings(t,MisStep[i,:,0]))
    MisEvents1Step.append(convert_to_timings(t,MisStep[i,:,0]))
    LearnedEvents1Step.append(convert_to_timings(t,LearnedStep[i,:,0]))
PrelearnEvents1Step.append(RefEvents1Step)

# Neuron 2 step events
RefEvents2Step = convert_to_timings(t,RefStep[:,1])
PrelearnEvents2Step     = [[]]
MisEvents2Step     = [[]] # Not used.
LearnedEvents2Step = [[]]
for i in range(num_trials):
    PrelearnEvents2Step.append(convert_to_timings(t,MisStep[i,:,1]))
    MisEvents2Step.append(convert_to_timings(t,MisStep[i,:,1]))
    LearnedEvents2Step.append(convert_to_timings(t,LearnedStep[i,:,1]))
PrelearnEvents2Step.append(RefEvents2Step)

# Colouring of plots
from itertools import repeat
mis_color = 'tab:blue'
color_list = []
color_list.extend(repeat(mis_color,num_trials+1))
color_list.append('tab:red') # Ref colour

## PLOTTING
# Fluctuating Input
dpi = 200
# fig, axs = plt.subplots(1,2)
fig = plt.figure(constrained_layout=True,dpi=dpi)
gs = GridSpec(5, 2, figure=fig, height_ratios=[1,0.6,1,1,1])
# Raster plots
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[4, :])
l1 = ax1.eventplot(PrelearnEvents1,colors=color_list)
l2 = ax1.eventplot(LearnedEvents1, colors='tab:orange', linelengths=0.7)
# ax1.set_yticks(np.arange(num_trials+1))
ax1.set_yticks([1,3,5])
ax1.set_ylabel("Trial")
ax1.axis(xmin=t[0]-1.25, xmax=t[-1]+1.25,ymin=0.2,ymax=num_trials+1.8)

l1 = ax2.eventplot(PrelearnEvents2,colors=color_list)
l2 = ax2.eventplot(LearnedEvents2, colors='tab:orange', linelengths=0.7)
ax2.set_yticks([1,3,5])
ax2.set_xlabel(r"t [$\times 10 ^3$]")
ax2.axis(xmin=t[0]-1.25, xmax=t[-1]+1.25,ymin=0.2,ymax=num_trials+1.8)
# plt.savefig("sec5_LR_II_raster.png")

# Current plot
axI = fig.add_subplot(gs[1,:])
axI.plot(t[:-1], noise)
axI.set_ylabel("I")
axI.set_xlabel(r"t [$\times 10 ^3$]")

# Time-series plot
blend = 0.5 # For alpha parameter of Line2D
num_samps = num_trials # Number of trials to plot.
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[3, 0])
ax6 = fig.add_subplot(gs[3, 1])
ax3.plot(t, Ref[:,0], color='tab:red')
for i in range(num_samps):
    ax3.plot(t, Mis[i,:,0], color='tab:blue', alpha=blend)
ax4.plot(t, Ref[:,0], color='tab:red')
for i in range(num_samps):
    ax4.plot(t, Learned[i,:,0], color='tab:orange', alpha=blend)
ax3.set_ylabel("V")

ax5.plot(t, Ref[:,1], color='tab:red')
for i in range(num_samps):
    ax5.plot(t, Mis[i,:,1], color='tab:blue')
ax6.plot(t, Ref[:,1], color='tab:red')
for i in range(num_samps):
    ax6.plot(t, Learned[i,:,1], color='tab:orange')
ax5.set_ylabel("V")
ax5.set_xlabel(r"t [$\times 10 ^3$]")
ax6.set_xlabel(r"t [$\times 10 ^3$]")
# plt.savefig("sec5_LR_II_example.png")
plt.savefig("sec5_LR_II.png")


# Step Input
figStep = plt.figure(constrained_layout=True,dpi=dpi)
gs = GridSpec(5, 2, figure=figStep, height_ratios=[1,0.6,1,1,1])
ax1 = figStep.add_subplot(gs[0, :])
ax2 = figStep.add_subplot(gs[4, :])
l1 = ax1.eventplot(PrelearnEvents1Step,colors=color_list)
l2 = ax1.eventplot(LearnedEvents1Step, colors='tab:orange', linelengths=0.7)
ax1.set_yticks([1,3,5])
ax1.set_ylabel("Trial")
ax1.axis(xmin=t[0]-1.25, xmax=t[-1]+1.25,ymin=0.2,ymax=num_trials+1.8)

l1 = ax2.eventplot(PrelearnEvents2Step,colors=color_list)
l2 = ax2.eventplot(LearnedEvents2Step, colors='tab:orange', linelengths=0.7)
ax2.set_yticks([1,3,5])
ax2.set_xlabel(r"t [$\times 10 ^3$]")
ax2.axis(xmin=t[0]-1.25, xmax=t[-1]+1.25,ymin=0.2,ymax=num_trials+1.8)

# Current plot
axI = figStep.add_subplot(gs[1,:])
Istep = 0.4*np.ones(int(25000/0.1+1))
Istep[:50000] = 0.2
axI.plot(t[:-1], noise)
axI.set_ylabel("I")
axI.set_xlabel(r"t [$\times 10 ^3$]")

# Time-series plot
ax3 = figStep.add_subplot(gs[2, 0])
ax4 = figStep.add_subplot(gs[2, 1])
ax5 = figStep.add_subplot(gs[3, 0])
ax6 = figStep.add_subplot(gs[3, 1])
ax3.plot(t, RefStep[:,0], color='tab:red')
for i in range(num_samps):
    ax3.plot(t, MisStep[i,:,0], color='tab:blue', alpha=blend)
ax4.plot(t, RefStep[:,0], color='tab:red')
for i in range(num_samps):
    ax4.plot(t, LearnedStep[i,:,0], color='tab:orange', alpha=blend)
ax3.set_ylabel("V")

ax5.plot(t, RefStep[:,1], color='tab:red')
for i in range(num_samps):
    ax5.plot(t, MisStep[i,:,1], color='tab:blue')
ax6.plot(t, RefStep[:,1], color='tab:red')
for i in range(num_samps):
    ax6.plot(t, LearnedStep[i,:,1], color='tab:orange')
ax5.set_ylabel("V")
ax5.set_xlabel(r"t [$\times 10 ^3$]")
ax6.set_xlabel(r"t [$\times 10 ^3$]")
plt.savefig("sec5_LR_II_step.png")

plt.show()
