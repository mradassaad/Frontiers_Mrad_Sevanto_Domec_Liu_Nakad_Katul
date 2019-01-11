import numpy as np
import pickle
import matplotlib.pyplot as plt

pickle_in = open("../Fig3/Fig3.resistant", "rb")
resistant = pickle.load(pickle_in)

pickle_in = open("../Fig3/Fig3.midrange", "rb")
midrange = pickle.load(pickle_in)

pickle_in = open("../Fig3/Fig3.vulnerable", "rb")
vulnerable = pickle.load(pickle_in)

unit = 1e3 / (3600*24)

# ------------TIME----------------

fig, ax = plt.subplots()
res_gl, = ax.plot(resistant["t"], resistant["gl"] * unit, 'k')
mid_gl, = ax.plot(midrange["t"], midrange["gl"] * unit, 'k--')
vul_gl, = ax.plot(vulnerable["t"], vulnerable["gl"] * unit, 'k:')

plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)

ax.set_xlabel("Time, $t$, days", FontSize=14)
ax.set_ylabel("Stomatal conductance, $g_s$, mmol m$^{-2}$ s$^{-1}$", FontSize=14)
ax.set_ylim(0, np.max(ax.yaxis.get_data_interval()))

noon = np.arange(24, np.alen(resistant["t"]), 48)

res_gl_day, = ax.plot(resistant["t"][noon], resistant["gl"][noon] * unit, 'r')
mid_gl_day, = ax.plot(midrange["t"][noon], midrange["gl"][noon] * unit, 'r--')
vul_gl_day, = ax.plot(vulnerable["t"][noon], vulnerable["gl"][noon] * unit, 'r:')

# legend1 = ax.legend((res_gl, mid_gl, vul_gl),
#                    ('$\psi_{63}=3$', '$\psi_{63}=2.2$', '$\psi_{63}=1.9$'), fontsize='large', loc=1, title='s=2')
# ax.add_artist(legend1)
legend1 = ax.legend((res_gl, res_gl_day),
                   ('Half-hourly', 'Midday'), fontsize='large', loc=1)

fig.savefig('../Fig3/gs_time.pdf', bbox_inches='tight')
# ------------psi_x----------------


fig2, ax2 = plt.subplots()
res_gl_day2, = ax2.plot(-resistant["psi_x"][noon], resistant["gl"][noon] * unit, 'r')
mid_gl_day2, = ax2.plot(-midrange["psi_x"][noon], midrange["gl"][noon] * unit, 'r--')
vul_gl_day2, = ax2.plot(-vulnerable["psi_x"][noon], vulnerable["gl"][noon] * unit, 'r:')

plt.setp(ax2.get_xticklabels(), FontSize=12)
plt.setp(ax2.get_yticklabels(), FontSize=12)

ax2.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
ax2.set_ylabel("Stomatal conductance, $g_s$, mmol m$^{-2}$ s$^{-1}$", FontSize=14)
ax2.set_ylim(0, np.max(ax.yaxis.get_data_interval()))

legend1 = ax2.legend((res_gl, mid_gl, vul_gl),
                   ('$\psi_{63}=3$', '$\psi_{63}=2.2$', '$\psi_{63}=1.9$'), fontsize='large', loc=2)

fig2.savefig('../Fig3/gs_psix.pdf', bbox_inches='tight')