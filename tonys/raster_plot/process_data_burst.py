import numpy as np
import matplotlib.pyplot as plt

data = np.load("sec4_CB_burst.npz")

noise = data['noise']
mis_arr = data['mis_arr']
mis_t_arr = data['mis_t_arr']

t = data['t']; Ref = data['Ref']; Mis = data['Mis']; Learned = data['Learned'];
thetalearned = data['thetalearned']

num_trials = Mis.shape[1]
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
    PrelearnEvents.append(convert_to_timings(t,Mis[:,i]))
    MisEvents.append(convert_to_timings(t,Mis[:,i]))
    LearnedEvents.append(convert_to_timings(t,Learned[:,i]))

PrelearnEvents.append(RefEvents)

# Colouring of plots
from itertools import repeat
mis_color = 'tab:blue'
color_list = []
color_list.extend(repeat(mis_color,num_trials+1))
color_list.append('tab:red') # Ref colour

# plt.eventplot(MisEvents)
fig, ax = plt.subplots(1,1)
l1 = ax.eventplot(PrelearnEvents,colors=color_list, lineoffsets=1)
l2 = ax.eventplot(LearnedEvents, colors='tab:orange', linelengths=0.7)
ax.set_yticks([1,2,3,4,5,6,7,8])
ax.axis(xmin=t[0], xmax=t[-1],ymin=0.2,ymax=9.8)
plt.xlabel("t / ms")
plt.ylabel("Trial")
plt.savefig("sec5_CB_bursting_raster.png")

# Example Plot
eg_sim = 3 # Which simulation to use for the example plot
fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
axs[0].plot(t, Ref, color='tab:red')
axs[0].plot(t, Mis[:,eg_sim], color='tab:blue')
axs[1].plot(t, Ref, color='tab:red')
axs[1].plot(t, Learned[:,eg_sim], color='tab:orange')
axs[0].set_xlabel("t / ms")
axs[1].set_xlabel("t / ms")
axs[0].set_ylabel("V / mV")
plt.savefig("sec5_CB_bursting_example.png")

plt.show()
