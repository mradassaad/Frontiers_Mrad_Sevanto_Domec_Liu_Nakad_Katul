import numpy as np
import pickle
import matplotlib.pyplot as plt
from Useful_Funcs import lam_from_trans
from Plant_Env_Props import ca

pickle_in = open("../Fig5/Fig5.resistant_WUS", "rb")
resistant = pickle.load(pickle_in)

pickle_in = open("../Fig5/Fig5.midrange_WUS", "rb")
midrange = pickle.load(pickle_in)

pickle_in = open("../Fig5/Fig5.vulnerable_WUS", "rb")
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

fig.savefig('../Fig5/gs_time_WUS.pdf', bbox_inches='tight')
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

fig2.savefig('../Fig5/gs_psix_WUS.pdf', bbox_inches='tight')

# --------------lambda-------------

pickle_in = open("../Fig5/Fig5.environment", "rb")
env = pickle.load(pickle_in)

pickle_in = open("../Fig5/Fig5.soil", "rb")
soil = pickle.load(pickle_in)

pickle_in = open("../Fig5/Fig5.plant", "rb")
plant = pickle.load(pickle_in)

t = resistant['t']

lam_low_resistant = lam_from_trans(plant['trans_max_interp'](resistant['psi_x']), ca, plant['alpha'],
                                   env['cp_interp'](t), env['VPDinterp'](t),
                                   env['k1_interp'](t), env['k2_interp'](t), plant['lai'])

lam_low_midrange = lam_from_trans(plant['trans_max_interp'](midrange['psi_x']), ca, plant['alpha'],
                                   env['cp_interp'](t), env['VPDinterp'](t),
                                   env['k1_interp'](t), env['k2_interp'](t), plant['lai'])

t = vulnerable['t']

lam_low_vulnerable = lam_from_trans(plant['trans_max_interp'](vulnerable['psi_x']), ca, plant['alpha'],
                                   env['cp_interp'](t), env['VPDinterp'](t),
                                   env['k1_interp'](t), env['k2_interp'](t), plant['lai'])

t = resistant['t']

lam_up_resistant = np.ones(lam_low_resistant.shape) * (ca - env['cp_interp'](t)) / \
                   env['VPDinterp'](t) / plant['alpha']


fig3, ax3 = plt.subplots()
res_lam, = ax3.plot(resistant["t"], resistant["lam"], 'k')
mid_lam, = ax3.plot(midrange["t"], midrange["lam"], 'k--')
vul_lam, = ax3.plot(vulnerable["t"], vulnerable["lam"], 'k:')

res_lam_low, = ax3.plot(resistant["t"], resistant["lam_low"], 'r')
res_lam_mid, = ax3.plot(midrange["t"], midrange["lam_low"], 'r--')
res_lam_vul, = ax3.plot(vulnerable["t"], vulnerable["lam_low"], 'r:')

plt.setp(ax3.get_xticklabels(), FontSize=12)
plt.setp(ax3.get_yticklabels(), FontSize=12)

ax3.set_xlabel("Time, $t$, days", FontSize=14)
ax3.set_ylabel("Marginal water use efficiency, $\lambda$, mol m$^{-2}$", FontSize=14)
# ax3.set_ylim(0, np.max(ax.yaxis.get_data_interval()))

legend1 = ax3.legend((res_lam, mid_lam, vul_lam),
                   ('$\psi_{63}=3$', '$\psi_{63}=2.2$', '$\psi_{63}=1.9$'), fontsize='large', loc=2)
ax3.add_artist(legend1)
legend2 = ax3.legend((res_lam, res_lam_low),
                   ('$\lambda (t)$', '$\lambda_{lower}$'), fontsize='large', loc=9)

fig3.savefig('../Fig5/lam_t_WUS.pdf', bbox_inches='tight')

# --------------------- PLC ---------------------

fig4, ax4 = plt.subplots()
res_PLC, = ax4.plot(resistant["t"], resistant["PLC"], 'k')
mid_PLC, = ax4.plot(midrange["t"], midrange["PLC"], 'k--')
vul_PLC, = ax4.plot(vulnerable["t"], vulnerable["PLC"], 'k:')

plt.setp(ax4.get_xticklabels(), FontSize=12)
plt.setp(ax4.get_yticklabels(), FontSize=12)

ax4.set_xlabel("Time, $t$, days", FontSize=14)
ax4.set_ylabel("Percent loss of conductivity, PLC, $\%$", FontSize=14)

legend1 = ax4.legend((res_PLC, mid_PLC, vul_PLC),
                   ('$\psi_{63}=3$', '$\psi_{63}=2.2$', '$\psi_{63}=1.9$'), fontsize='large', loc=2)
ax4.set_ylim(0, 100)

fig4.savefig('../Fig5/PLC_t_WUS.pdf', bbox_inches='tight')