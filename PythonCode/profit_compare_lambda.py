import numpy as np
import pickle
import matplotlib.pyplot as plt
from Useful_Funcs import lam_from_trans
# from Plant_Env_Props import ca

pickle_in = open("../profit_compare/profit_compare.16percent_9lambda", "rb")
vulnerable = pickle.load(pickle_in)

pickle_in = open("../profit_compare/profit_compare.16percent_10lambda", "rb")
vulnerable2 = pickle.load(pickle_in)

pickle_in = open("../profit_compare/profit_compare.16percent_105lambda", "rb")
vulnerable3 = pickle.load(pickle_in)

pickle_in = open("../profit_compare/profit_compare.16percent_11lambda", "rb")
vulnerable4 = pickle.load(pickle_in)

unit = 1e3 / (3600*24)
sun = np.tile(np.arange(12, 36, 1), (10, 1)) + np.transpose(np.tile(np.arange(0, 10, 1), (24, 1))) * 48
sun = sun.flatten()
full_interval = np.arange(0, vulnerable["t"].shape[0], 1)
noon = np.arange(24, np.alen(vulnerable["t"]), 48)

# ------------ g_s_psix noon ----------------
pickle_in = open("../no_WUS/environment", "rb")
env = pickle.load(pickle_in)

xax = "psi_x"
division = noon
flip_sign = -1

fig2, ax2 = plt.subplots()
vul_gl_noon, = ax2.plot(flip_sign * vulnerable[xax][division], vulnerable["gl"][division] * unit, 'k-')
vul_gl_noon2, = ax2.plot(flip_sign * vulnerable2[xax][division], vulnerable2["gl"][division] * unit, 'k--')
vul_gl_noon3, = ax2.plot(flip_sign * vulnerable3[xax][division], vulnerable3["gl"][division] * unit, 'k:')
vul_gl_noon4, = ax2.plot(flip_sign * vulnerable4[xax][division], vulnerable4["gl"][division] * unit, 'k-.')


vul_gl_profit_noon, = ax2.plot(flip_sign * vulnerable2[xax][division],
                          vulnerable2["E_opt"][division] / 1.6 / env["VPDinterp"](vulnerable["t"][division]) * unit, 'r-')


plt.setp(ax2.get_xticklabels(), FontSize=12)
plt.setp(ax2.get_yticklabels(), FontSize=12)

# ax2.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
ax2.set_ylabel("Midday stomatal conductance, $g_s$, mmol m$^{-2}$ s$^{-1}$", FontSize=11)
ax2.set_ylim(5, 45)
ax2.set_xlim(-0.45, -0.15)

legend1 = ax2.legend((vul_gl_noon, vul_gl_noon2, vul_gl_noon3, vul_gl_noon4, vul_gl_profit_noon),
                   ('9 mmol mol$^{-1}$', '10 mmol mol$^{-1}$',
                    '10.5 mmol mol$^{-1}$', '11 mmol mol$^{-1}$', 'profit maximization'), fontsize='large', loc=2)

# fig2.savefig('../profit_compare/lambda_noon.pdf', bbox_inches='tight')

# ------------ g_s_psix morning ----------------
pickle_in = open("../no_WUS/environment", "rb")
env = pickle.load(pickle_in)
morning = np.arange(16, np.alen(vulnerable["t"]), 48)
xax = "psi_x"
division = morning
flip_sign = -1

fig_morning, ax_morning = plt.subplots()
vul_gl_morning, = ax_morning.plot(flip_sign * vulnerable[xax][division], vulnerable["gl"][division] * unit, 'k-')
vul_gl_morning, = ax_morning.plot(flip_sign * vulnerable2[xax][division], vulnerable2["gl"][division] * unit, 'k--')
vul_gl_morning, = ax_morning.plot(flip_sign * vulnerable3[xax][division], vulnerable3["gl"][division] * unit, 'k:')
vul_gl_morning, = ax_morning.plot(flip_sign * vulnerable4[xax][division], vulnerable4["gl"][division] * unit, 'k-.')


vul_gl_profit_morning, = ax_morning.plot(flip_sign * vulnerable2[xax][division],
                          vulnerable2["E_opt"][division] / 1.6 / env["VPDinterp"](vulnerable["t"][division]) * unit, 'r-')


plt.setp(ax_morning.get_xticklabels(), FontSize=12)
plt.setp(ax_morning.get_yticklabels(), FontSize=12)

# ax2.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
ax_morning.set_ylabel("Morning stomatal conductance, $g_s$, mmol m$^{-2}$ s$^{-1}$", FontSize=11)
ax_morning.set_ylim(5, 45)
ax_morning.set_xlim(-0.45, -0.15)

# legend1 = ax2.legend((res_gl, mid_gl, vul_gl),
#                    ('resistant', 'exponential', 'vulnerable'), fontsize='large', loc=2)

# fig_morning.savefig('../profit_compare/lambda_morning.pdf', bbox_inches='tight')

# ------------ g_s_psix morning ----------------
pickle_in = open("../no_WUS/environment", "rb")
env = pickle.load(pickle_in)
evening = np.arange(36, np.alen(vulnerable["t"]), 48)
xax = "psi_x"
division = evening
flip_sign = -1

fig_evening, ax_evening = plt.subplots()
vul_gl_evening, = ax_evening.plot(flip_sign * vulnerable[xax][division], vulnerable["gl"][division] * unit, 'k-')
vul_gl_evening, = ax_evening.plot(flip_sign * vulnerable2[xax][division], vulnerable2["gl"][division] * unit, 'k--')
vul_gl_evening, = ax_evening.plot(flip_sign * vulnerable3[xax][division], vulnerable3["gl"][division] * unit, 'k:')
vul_gl_evening, = ax_evening.plot(flip_sign * vulnerable4[xax][division], vulnerable4["gl"][division] * unit, 'k-.')


vul_gl_profit_evening, = ax_evening.plot(flip_sign * vulnerable2[xax][division],
                          vulnerable2["E_opt"][division] / 1.6 / env["VPDinterp"](vulnerable["t"][division]) * unit, 'r-')


plt.setp(ax_evening.get_xticklabels(), FontSize=12)
plt.setp(ax_evening.get_yticklabels(), FontSize=12)

ax_evening.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
ax_evening.set_ylabel("Evening stomatal conductance, $g_s$, mmol m$^{-2}$ s$^{-1}$", FontSize=11)
ax_evening.set_ylim(5, 45)
ax_evening.set_xlim(-0.45, -0.15)

# legend1 = ax2.legend((res_gl, mid_gl, vul_gl),
#                    ('resistant', 'exponential', 'vulnerable'), fontsize='large', loc=2)

# fig_evening.savefig('../profit_compare/lambda_evening.pdf', bbox_inches='tight')