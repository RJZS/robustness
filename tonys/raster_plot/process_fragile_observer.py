import numpy as np
import matplotlib.pyplot as plt

data = np.load("sec3_fragile_observer.npz")

mis_arr = data['mis_arr']
mis_t_arr = data['mis_t_arr']

t=data['t']
thetalearned = data['thetalearned']

print(thetalearned.shape)
# for i in range(7):
#     plt.figure()
#     plt.plot(t, thetalearned[i,0,:], t, thetalearned[i,1,:])
#     plt.ylabel(i)

# fig, axs = plt.subplots(1, 2)
# axs[0].plot(t, thetalearned[2,0,:], t, thetalearned[2,1,:])
# axs[0].plot(t, thetalearned[3,0,:], t, thetalearned[3,1,:])
plt.plot(t, thetalearned[0,0,:],color=u'#1f77b4')
plt.plot(t, thetalearned[6,0,:],color=u'#2ca02c')
plt.plot(t, thetalearned[3,0,:],color=u'#9467bd')
plt.plot(t, thetalearned[0,1,:],color=u'#ff7f0e')
plt.plot(t, thetalearned[6,1,:],color=u'#d62728')
plt.plot(t, thetalearned[3,1,:],color=u'#8c564b')
plt.xlabel("t / ms")
plt.ylabel(r'$\mu_x$ / mS $cm^{-2}$')
plt.legend([r'$\hat{\mu}_{\rm{Na}}$',r'$\hat{\mu}_{\rm{KCa}}$',r'$\hat{\mu}_{\rm{A}}$'])

plt.savefig("sec3_fragile_nondiag_observer.pdf")
plt.show()

