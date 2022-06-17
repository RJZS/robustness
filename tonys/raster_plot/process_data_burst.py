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
PrelearnEvents     = []
MisEvents     = [] # Not used.
LearnedEvents = []
for i in range(num_trials):
    PrelearnEvents.append(convert_to_timings(t,Mis[:,i]))
    MisEvents.append(convert_to_timings(t,Mis[:,i]))
    LearnedEvents.append(convert_to_timings(t,Learned[:,i]))

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
