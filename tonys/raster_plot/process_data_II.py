import numpy as np
import matplotlib.pyplot as plt

data = np.load("sec4_CB_II.npz")

noise = data['noise']
mis_arr = data['mis_arr']
mis_t_arr = data['mis_t_arr']

t = data['t']; Ref = data['Ref']; Mis = data['Mis']; Learned = data['Learned'];
thetalearned = data['thetalearned']

num_trials = Mis.shape[2]

def convert_to_timings(t, y, thresh=0):
    event_idxs = np.where(y>thresh)
    event_times = t[event_idxs]
    # Use np.diffs to eliminate all but the first spike in each burst.
    return event_times

# Neuron 1 events
RefEvents1 = convert_to_timings(t,Ref[0,:])
PrelearnEvents1     = []
MisEvents1     = [] # Not used.
LearnedEvents1 = []
for i in range(num_trials):
    PrelearnEvents1.append(convert_to_timings(t,Mis[0,:,i]))
    MisEvents1.append(convert_to_timings(t,Mis[0,:,i]))
    LearnedEvents1.append(convert_to_timings(t,Learned[0,:,i]))

PrelearnEvents1.append(RefEvents1)

# Neuron 2 events
RefEvents2 = convert_to_timings(t,Ref[1,:])
PrelearnEvents2     = []
MisEvents2     = [] # Not used.
LearnedEvents2 = []
for i in range(num_trials):
    PrelearnEvents2.append(convert_to_timings(t,Mis[1,:,i]))
    MisEvents2.append(convert_to_timings(t,Mis[1,:,i]))
    LearnedEvents2.append(convert_to_timings(t,Learned[1,:,i]))

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

plt.figure()
plt.eventplot([RefEvents1, RefEvents2])

plt.figure()
plt.plot(t,Mis[0,:,0], t, Mis[1,:,0])

plt.show()
