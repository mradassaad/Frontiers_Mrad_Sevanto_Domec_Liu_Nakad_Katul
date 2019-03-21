import numpy as np
import pickle
import matplotlib.pyplot as plt
from Useful_Funcs import lam_from_trans
# from Plant_Env_Props import ca

pickle_in = open("../WUS_no_comp/result.resistant", "rb")
resistant = pickle.load(pickle_in)

pickle_in = open("../WUS_no_comp/result.low_s", "rb")
exponential = pickle.load(pickle_in)

pickle_in = open("../WUS_no_comp/result.vulnerable", "rb")
vulnerable = pickle.load(pickle_in)

pickle_in = open("../WUS_comp/result.resistant", "rb")
resistant_comp = pickle.load(pickle_in)

pickle_in = open("../WUS_comp/result.low_s", "rb")
exponential_comp = pickle.load(pickle_in)

pickle_in = open("../WUS_comp/result.vulnerable", "rb")
vulnerable_comp = pickle.load(pickle_in)

noon = np.arange(24, np.alen(resistant["t"]), 48)
unit = 1e3 / (3600*24)
ca = 350e-6  # mol mol-1

xax = "psi_x"
division = noon
flip_sign = -1

fig, ax = plt.subplots()
res_cica, = ax.plot(flip_sign * resistant[xax][division], resistant["ci"][division] / ca, 'k')
mid_cica, = ax.plot(flip_sign * exponential[xax][division], exponential["ci"][division] / ca, 'k--')
vul_cica, = ax.plot(flip_sign * vulnerable[xax][division], vulnerable["ci"][division] / ca, 'k:')


plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)

# ax.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
ax.set_ylabel("Midday ratio of internal to atmospheric carbon concentration, $c_i / c_a$", FontSize=9)
ax.set_ylim(0.2, 0.7)
legend1 = ax.legend((res_cica, mid_cica, vul_cica),
                    ('Steep and resistant', 'Gradual and resistant', 'Steep and vulnerable'),
                    fontsize='large', loc=4)

# fig.savefig('../PLC_cica/cica.pdf', bbox_inches='tight')


fig2, ax2 = plt.subplots()
res_PLC, = ax2.plot(flip_sign * resistant[xax][division], resistant["PLC"][division], 'k')
mid_PLC, = ax2.plot(flip_sign * exponential[xax][division], exponential["PLC"][division], 'k--')
vul_PLC, = ax2.plot(flip_sign * vulnerable[xax][division], vulnerable["PLC"][division], 'k:')


plt.setp(ax2.get_xticklabels(), FontSize=12)
plt.setp(ax2.get_yticklabels(), FontSize=12)

ax2.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
ax2.set_ylabel("Midday percent loss of conductance, PLC, %", FontSize=11)
ax2.set_xlim(- 0.5, 0)
ax2.set_ylim(0, 100)
# legend1 = ax2.legend((res_cica, mid_cica, vul_cica),
#                     ('Steep and resistant', 'Gradual and resistant', 'Steep and vulnerable'),
#                     fontsize='large', loc=4)

# fig2.savefig('../PLC_cica/PLC.pdf', bbox_inches='tight')


fig3, ax3 = plt.subplots()
res_cica_comp, = ax3.plot(flip_sign * resistant_comp[xax][division], resistant_comp["ci"][division] / ca, 'k')
mid_cica_comp, = ax3.plot(flip_sign * exponential_comp[xax][division], exponential_comp["ci"][division] / ca, 'k--')
vul_cica_comp, = ax3.plot(flip_sign * vulnerable_comp[xax][division], vulnerable_comp["ci"][division] / ca, 'k:')


plt.setp(ax3.get_xticklabels(), FontSize=12)
plt.setp(ax3.get_yticklabels(), FontSize=12)
ax3.set_ylim(0.2, 0.7)
# ax3.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
# ax3.set_ylabel("Midday ratio of internal to atmospheric carbon concentration, $c_i / c_a$", FontSize=9)
# #
# legend1 = ax3.legend((res_cica, mid_cica, vul_cica),
#                     ('Steep and resistant', 'Gradual and resistant', 'Steep and vulnerable'),
#                     fontsize='large', loc=4)

# fig3.savefig('../PLC_cica/cica_comp.pdf', bbox_inches='tight')

fig4, ax4 = plt.subplots()
res_PLC_comp, = ax4.plot(flip_sign * resistant_comp[xax][division], resistant_comp["PLC"][division], 'k')
mid_PLC_comp, = ax4.plot(flip_sign * exponential_comp[xax][division], exponential_comp["PLC"][division], 'k--')
vul_PLC_comp, = ax4.plot(flip_sign * vulnerable_comp[xax][division], vulnerable_comp["PLC"][division], 'k:')


plt.setp(ax4.get_xticklabels(), FontSize=12)
plt.setp(ax4.get_yticklabels(), FontSize=12)

ax4.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=14)
# ax4.set_ylabel("Midday percent loss of conductance, PLC, %", FontSize=11)
ax4.set_xlim(- 0.5, 0)
ax4.set_ylim(0, 100)
# legend1 = ax4.legend((res_cica, mid_cica, vul_cica),
#                     ('Steep and resistant', 'Gradual and resistant', 'Steep and vulnerable'),
#                     fontsize='large', loc=4)

# fig4.savefig('../PLC_cica/PLC_comp.pdf', bbox_inches='tight')
