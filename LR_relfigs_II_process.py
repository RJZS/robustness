## Generate the raster plots for the reliability experiment
# on Luka's bursting circuit, using data generated in 'LR_relfigs_burst_gen.jl'.

import matplotlib.pyplot as plt
import numpy as np
import h5py

# fname = "sec4_LR_II_oldpaperversion.jld" # Version previously used in paper.
fname = "sec4_LR_II_noinact_withstep.jld"

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

# plt.eventplot(MisEvents)
fig, axs = plt.subplots(1,2)
l1 = axs[0].eventplot(PrelearnEvents1,colors=color_list)
l2 = axs[0].eventplot(LearnedEvents1, colors='tab:orange', linelengths=0.7)
axs[0].set_yticks(np.arange(num_trials))
axs[0].set_ylabel("Trial")
axs[0].set_xlabel("t")
axs[0].axis(xmin=t[0], xmax=t[-1],ymin=0.2,ymax=num_trials+1.8)

l1 = axs[1].eventplot(PrelearnEvents2,colors=color_list)
l2 = axs[1].eventplot(LearnedEvents2, colors='tab:orange', linelengths=0.7)
axs[1].set_yticks(np.arange(num_trials))
axs[1].set_xlabel("t")
axs[1].axis(xmin=t[0], xmax=t[-1],ymin=0.2,ymax=num_trials+1.8)
plt.savefig("sec5_LR_II_raster.png")

# Example plot
eg_sim = 0 # Which simulation to use for the example plot
fig, axs = plt.subplots(2, 2, sharey=True)
axs[0][0].plot(t, Ref[:,0], color='tab:red')
axs[0][0].plot(t, Mis[eg_sim,:,0], color='tab:blue')
axs[0][1].plot(t, Ref[:,0], color='tab:red')
axs[0][1].plot(t, Learned[eg_sim,:,0], color='tab:orange')
axs[0][0].set_ylabel("V")

axs[1][0].plot(t, Ref[:,1], color='tab:red')
axs[1][0].plot(t, Mis[eg_sim,:,1], color='tab:blue')
axs[1][1].plot(t, Ref[:,1], color='tab:red')
axs[1][1].plot(t, Learned[eg_sim,:,1], color='tab:orange')
axs[1][0].set_ylabel("V")
axs[1][0].set_xlabel("t")
axs[1][1].set_xlabel("t")
plt.savefig("sec5_LR_II_example.png")

# Raster for step input
fig, axs = plt.subplots(1,2)
l1 = axs[0].eventplot(PrelearnEvents1Step,colors=color_list)
l2 = axs[0].eventplot(LearnedEvents1Step, colors='tab:orange', linelengths=0.7)
axs[0].set_yticks(np.arange(num_trials))
axs[0].set_ylabel("Trial")
axs[0].set_xlabel("t")
axs[0].axis(xmin=t[0], xmax=t[-1],ymin=0.2,ymax=num_trials+1.8)

l1 = axs[1].eventplot(PrelearnEvents2Step,colors=color_list)
l2 = axs[1].eventplot(LearnedEvents2Step, colors='tab:orange', linelengths=0.7)
axs[1].set_yticks(np.arange(num_trials))
axs[1].set_xlabel("t")
axs[1].axis(xmin=t[0], xmax=t[-1],ymin=0.2,ymax=num_trials+1.8)
plt.savefig("sec5_LR_II_raster_step.png")

plt.show()
