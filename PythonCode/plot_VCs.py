import numpy as np
import matplotlib.pyplot as plt

Kmax = 2  # mmol m-2 s-1 for a tree that has 20 mmol m-1 s-1 conductivity and 10 meters long
psi_63 = 4.3  # MPa
w_exp = 5.4
psi = np.linspace(0, 8, 100)

VC_steep_resistant = Kmax * np.exp(- (psi / psi_63) ** w_exp)

psi_63 = 4.3  # MPa
w_exp = 2

VC_gradual_resistant = Kmax * np.exp(- (psi / psi_63) ** w_exp)

psi_63 = 1.5  # MPa
w_exp = 5.4

VC_steep_vulnerable = Kmax * np.exp(- (psi / psi_63) ** w_exp)

fig, ax = plt.subplots()

steep_res, = plt.plot(-psi, VC_steep_resistant, 'k')
grad_res, = plt.plot(-psi, VC_gradual_resistant, 'k--')
steep_vul, = plt.plot(-psi, VC_steep_vulnerable, 'k:')

plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)

ax.set_xlabel("Water potential, $\psi$, MPa", FontSize=14)
ax.set_ylabel("Root-leaf conductance, $g_{rl}$, mmol m$^{-2}$ s$^{-1}$", FontSize=12)

legend1 = ax.legend((steep_res, grad_res, steep_vul),
                   ('Steep and resistant', 'Gradual and resistant', 'Steep and vulnerable'),
                     fontsize='large', loc=2, title="Vulnerability curve description")
fig.savefig('../grl.pdf', bbox_inches='tight')
