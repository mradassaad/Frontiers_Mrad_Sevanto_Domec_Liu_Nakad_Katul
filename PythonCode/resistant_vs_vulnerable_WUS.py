import numpy as np
import pickle
import matplotlib.pyplot as plt
from Useful_Funcs import lam_from_trans
# from Plant_Env_Props import ca

pickle_in = open("../WUS_no_comp/WUS_no_comp.resistant", "rb")
resistant = pickle.load(pickle_in)

pickle_in = open("../WUS_no_comp/WUS_no_comp.exponential", "rb")
exponential = pickle.load(pickle_in)

pickle_in = open("../WUS_no_comp/WUS_no_comp.vulnerable", "rb")
vulnerable = pickle.load(pickle_in)

unit = 1e3 / (3600*24)
sun = np.tile(np.arange(12, 36, 1), (10, 1)) + np.transpose(np.tile(np.arange(0, 10, 1), (24, 1))) * 48
sun = sun.flatten()
full_interval = np.arange(0, resistant["t"].shape[0], 1)

# ------------ gs_TIME ----------------

fig, ax = plt.subplots()
res_gl, = ax.plot(resistant["t"], resistant["gl"] * unit, 'k')
mid_gl, = ax.plot(exponential["t"], exponential["gl"] * unit, 'k--')
vul_gl, = ax.plot(vulnerable["t"], vulnerable["gl"] * unit, 'k:')

plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)

ax.set_xlabel("Time, $t$, days", FontSize=14)
ax.set_ylabel("Stomatal conductance, $g_s$, mmol m$^{-2}$ s$^{-1}$", FontSize=10)
ax.set_ylim(0, np.max(ax.yaxis.get_data_interval()))

noon = np.arange(24, np.alen(resistant["t"]), 48)

res_gl_day, = ax.plot(resistant["t"][noon], resistant["gl"][noon] * unit, 'r')
mid_gl_day, = ax.plot(exponential["t"][noon], exponential["gl"][noon] * unit, 'r--')
vul_gl_day, = ax.plot(vulnerable["t"][noon], vulnerable["gl"][noon] * unit, 'r:')

# legend1 = ax.legend((res_gl, mid_gl, vul_gl),
#                    ('$\psi_{63}=3$', '$\psi_{63}=2.2$', '$\psi_{63}=1.9$'), fontsize='large', loc=1, title='s=2')
# ax.add_artist(legend1)
legend1 = ax.legend((res_gl, res_gl_day),
                   ('Half-hourly', 'Midday'), fontsize='large', loc=1)
fig.set_figheight(3)
fig.set_figwidth(8.5)
# fig.savefig('../WUS_no_comp/gs_time.pdf', bbox_inches='tight')
# ------------ g_s_psix ----------------
pickle_in = open("../no_WUS/environment", "rb")
env = pickle.load(pickle_in)

xax = "psi_x"
division = noon
flip_sign = -1

fig2, ax2 = plt.subplots()
res_gl_day2, = ax2.plot(flip_sign * resistant[xax][division], resistant["gl"][division] * unit, 'k')
mid_gl_day2, = ax2.plot(flip_sign * exponential[xax][division], exponential["gl"][division] * unit, 'k--')
vul_gl_day2, = ax2.plot(flip_sign * vulnerable[xax][division], vulnerable["gl"][division] * unit, 'k:')

# res_gl_profit, = ax2.plot(flip_sign * resistant[xax][division],
#                           resistant["E_opt"][division] / 1.6 / env["VPDinterp"](resistant["t"][division]) * unit, 'r')
# mid_gl_profit, = ax2.plot(flip_sign * exponential[xax][division],
#                           exponential["E_opt"][division] / 1.6 / env["VPDinterp"](exponential["t"][division]) * unit, 'r--')
# vul_gl_profit, = ax2.plot(flip_sign * vulnerable[xax][division],
#                           vulnerable["E_opt"][division] / 1.6 / env["VPDinterp"](vulnerable["t"][division]) * unit, 'r:')


plt.setp(ax2.get_xticklabels(), FontSize=12)
plt.setp(ax2.get_yticklabels(), FontSize=12)

ax2.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
ax2.set_ylabel("Midday stomatal conductance, $g_s$, mmol m$^{-2}$ s$^{-1}$", FontSize=11)
# ax2.set_ylim(0, np.max(ax.yaxis.get_data_interval()))

legend1 = ax2.legend((res_gl, mid_gl, vul_gl),
                   ('resistant', 'exponential', 'vulnerable'), fontsize='large', loc=2)

# fig2.savefig('../WUS_no_comp/gs_psix.pdf', bbox_inches='tight')

# --------------lambda-------------

pickle_in = open("../no_WUS/soil", "rb")
soil = pickle.load(pickle_in)

pickle_in = open("../no_WUS/plant_resistant", "rb")
plant = pickle.load(pickle_in)

xax = "t"
division = full_interval
flip_sign = 1

lam_res = np.amax(np.array((resistant["lam"], resistant["lam_low"])), axis=0) * 1e3
lam_exp = np.amax(np.array((exponential["lam"], exponential["lam_low"])), axis=0) * 1e3
lam_vul = np.amax(np.array((vulnerable["lam"], vulnerable["lam_low"])), axis=0) * 1e3

#
# lam_up_resistant = np.ones(lam_low_resistant.shape) * (ca - env['cp_interp'](t)) / \
#                    env['VPDinterp'](t) / plant['alpha']

xax = "t"
fig3, ax3 = plt.subplots()
res_lam, = ax3.plot(resistant[xax], lam_res, 'k')
mid_lam, = ax3.plot(exponential[xax], lam_exp, 'k--')
vul_lam, = ax3.plot(vulnerable[xax], lam_vul, 'k:')

# res_lam_low, = ax3.plot(resistant["t"], resistant["lam_low"], 'r')
# res_lam_mid, = ax3.plot(midrange["t"], midrange["lam_low"], 'r--')
# res_lam_vul, = ax3.plot(vulnerable["t"], vulnerable["lam_low"], 'r:')

plt.setp(ax3.get_xticklabels(), FontSize=12)
plt.setp(ax3.get_yticklabels(), FontSize=12)

ax3.set_xlabel("Time, $t$, days", FontSize=14)
ax3.set_ylabel("Marginal water use efficiency, $\lambda$, mmol mol$^{-1}$", FontSize=10)
# ax3.set_ylim(0, np.max(ax.yaxis.get_data_interval()))

legend1 = ax3.legend((res_lam, mid_lam, vul_lam),
                   ('resistant', 'exponential', 'vulnerable'), fontsize='large', loc=2)
# ax3.add_artist(legend1)
# legend2 = ax3.legend((res_lam, res_lam_low),
#                    ('$\lambda (t)$', '$\lambda_{lower}$'), fontsize='large', loc=9)
fig3.set_figheight(3)
fig3.set_figwidth(8.5)
# # fig3.savefig('../WUS_no_comp/lam_t.pdf', bbox_inches='tight')

# ----------------------- A -----------------------

unit0 = 24 * 3600   # 1/s -> 1/d
xax = "t"
division = full_interval
flip_sign = 1

fig4, ax4 = plt.subplots()
res_A, = ax4.plot(flip_sign * resistant[xax][division],
                  resistant["A_val"][division] * 1e6 / unit0, 'k')
exp_A, = ax4.plot(flip_sign * exponential[xax][division],
                  exponential["A_val"][division] * 1e6 / unit0, 'k--')
vul_A, = ax4.plot(flip_sign * vulnerable[xax][division],
                  vulnerable["A_val"][division] * 1e6 / unit0, 'k:')

# res_A_opt, = ax4.plot(flip_sign * resistant[xax][division],
#                       resistant["A_opt"][division] * 1e6 / unit0, 'r')
# exp_A_opt, = ax4.plot(flip_sign * exponential[xax][division],
#                       exponential["A_opt"][division] * 1e6 / unit0, 'r--')
# vul_A_opt, = ax4.plot(flip_sign * vulnerable[xax][division],
#                       vulnerable["A_opt"][division] * 1e6 / unit0, 'r:')

plt.setp(ax4.get_xticklabels(), FontSize=12)
plt.setp(ax4.get_yticklabels(), FontSize=12)

ax4.set_xlabel("Time, $t$, days", FontSize=14)
ax4.set_ylabel("Carbon assimilation rate, $A$, $\mu$mol m$^{-2}$ s$^{-1}$", FontSize=12)
# ax3.set_ylim(0, np.max(ax.yaxis.get_data_interval()))
# fig4.savefig('../WUS_no_comp/A_t.pdf', bbox_inches='tight')
# -----------------------  E -----------------------

xax = "t"
division = full_interval
flip_sign = 1

fig5, ax5 = plt.subplots()
res_A, = ax5.plot(flip_sign * resistant[xax][division],
                  resistant["E"][division] * 1e3 / unit0, 'k')
exp_A, = ax5.plot(flip_sign * exponential[xax][division],
                  exponential["E"][division] * 1e3 / unit0, 'k--')
vul_A, = ax5.plot(flip_sign * vulnerable[xax][division],
                  vulnerable["E"][division] * 1e3 / unit0, 'k:')

# res_A_opt, = ax5.plot(flip_sign * resistant[xax][division],
#                       resistant["E_opt"][division] * 1e3 / unit0, 'r')
# exp_A_opt, = ax5.plot(flip_sign * exponential[xax][division],
#                       exponential["E_opt"][division] * 1e3 / unit0, 'r--')
# vul_A_opt, = ax5.plot(flip_sign * vulnerable[xax][division],
#                       vulnerable["E_opt"][division] * 1e3 / unit0, 'r.')

plt.setp(ax5.get_xticklabels(), FontSize=12)
plt.setp(ax5.get_yticklabels(), FontSize=12)

ax5.set_xlabel("Time, $t$, days", FontSize=14)
ax5.set_ylabel("Transpiration rate, $E$, mmol m$^{-2}$ s$^{-1}$", FontSize=12)

# fig5.savefig('../WUS_no_comp/E_t.pdf', bbox_inches='tight')
# ----------------------- psi_l -----------------------
xax = "psi_x"
division = noon
flip_sign = -1

fig6, ax6 = plt.subplots()
res_psil, = ax6.plot(flip_sign * resistant[xax][division],
                     -resistant["psi_l"][division], 'k')
exp_psil, = ax6.plot(flip_sign * exponential[xax][division],
                     -exponential["psi_l"][division], 'k--')
vul_psil, = ax6.plot(flip_sign * vulnerable[xax][division],
                     -vulnerable["psi_l"][division], 'k:')
#
# res_psil_opt, = ax6.plot(flip_sign * resistant[xax][division],
#                          -resistant["P_opt"][division], 'r')
# exp_psil_opt, = ax6.plot(flip_sign * exponential[xax][division],
#                          -exponential["P_opt"][division], 'r--')
# vul_psil_opt, = ax6.plot(flip_sign * vulnerable[xax][division],
#                          -vulnerable["P_opt"][division], 'r*')

plt.setp(ax6.get_xticklabels(), FontSize=12)
plt.setp(ax6.get_yticklabels(), FontSize=12)

ax6.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
ax6.set_ylabel("Midday leaf water potential, $\psi_l$, MPa", FontSize=12)

# fig6.savefig('../WUS_no_comp/psil_psix.pdf', bbox_inches='tight')
# -------------------- psi_x ---------
xax = "t"
division = full_interval
flip_sign = 1

fig7, ax7 = plt.subplots()
res_psix, = ax7.plot(flip_sign * resistant[xax][division],
                     -resistant["psi_x"][division], 'k')
exp_psix, = ax7.plot(flip_sign * exponential[xax][division],
                     -exponential["psi_x"][division], 'k--')
vul_psix, = ax7.plot(flip_sign * vulnerable[xax][division],
                     -vulnerable["psi_x"][division], 'k:')

plt.setp(ax7.get_xticklabels(), FontSize=12)
plt.setp(ax7.get_yticklabels(), FontSize=12)

ax7.set_xlabel("Time, $t$, days", FontSize=14)
ax7.set_ylabel("Soil water potential, $\psi_x$, MPa", FontSize=12)

# # --------------------- PLC ---------------------
#
# fig4, ax4 = plt.subplots()
# res_PLC, = ax4.plot(resistant["t"], resistant["PLC"], 'k')
# mid_PLC, = ax4.plot(midrange["t"], midrange["PLC"], 'k--')
# vul_PLC, = ax4.plot(vulnerable["t"], vulnerable["PLC"], 'k:')
#
# plt.setp(ax4.get_xticklabels(), FontSize=12)
# plt.setp(ax4.get_yticklabels(), FontSize=12)
#
# ax4.set_xlabel("Time, $t$, days", FontSize=14)
# ax4.set_ylabel("Percent loss of conductivity, PLC, $\%$", FontSize=14)
#
# legend1 = ax4.legend((res_PLC, mid_PLC, vul_PLC),
#                    ('$\psi_{63}=3$', '$\psi_{63}=2.2$', '$\psi_{63}=1.9$'), fontsize='large', loc=2)
# ax4.set_ylim(0, 100)

# fig4.savefig('../Fig3/PLC_t.pdf', bbox_inches='tight')


# #---------------------- ci/ca ---------------------
ca = 350e-6
ci_res = np.ones(resistant['t'].shape) * ca
notZero = resistant['gl'] != 0
ci_res[notZero] = ca - resistant['A_val'][notZero] / resistant['gl'][notZero]  # internal carbon concentration, mol/mol

plt.plot(resistant['t'], ci_res/ca)