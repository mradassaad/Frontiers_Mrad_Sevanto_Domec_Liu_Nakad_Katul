from Plant_Env_Props import *
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 24, np.alen(VPDavg))
#
# fig, ax = plt.subplots()
# VPD, = ax.plot(t, VPDavg, 'k')
#
# plt.setp(ax.get_xticklabels(), FontSize=12)
# plt.setp(ax.get_yticklabels(), FontSize=12)
#
# ax.set_xlabel("Time, $t$, hours", FontSize=14)
# ax.set_ylabel("Vapor pressure deficit, VPD, mol mol$^{-1}$", FontSize=14)
#
# axTemp = ax.twinx()
# axTemp.plot(t, TEMPavg, 'k--')
# axTemp.set_ylabel("Temperature, $T$, C", FontSize=14)
# plt.setp(axTemp.get_yticklabels(), FontSize=12)
#
# axPAR = ax.twinx()
# axPAR.plot(t, PARavg, 'k:')
# axPAR.

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par2,
                                    offset=(offset, 0))

par2.axis["right"].toggle(all=True)
par1.axis["right"].toggle(all=True)

# host.set_xlim(0, 2)
# host.set_ylim(0, 2)

host.set_xlabel("Time, $t$, hours", FontSize=14)
host.set_ylabel("Vapor pressure deficit, VPD, mmol mol$^{-1}$")
par1.set_ylabel("Temperature, $T_a$, K")
par2.set_ylabel("Photosynthetically active radiation, PAR, $\mu$mol m$^{-2}$ s$^{-1}$ ")

VPD, = host.plot(t, VPDavg * 1e3, 'k')
TEMP, = par1.plot(t, TEMPavg, 'k--')
PAR, = par2.plot(t, PARavg, 'k:')

# par1.set_ylim(270, 300)
# par2.set_ylim(1, 65)

host.legend((VPD, TEMP, PAR),
                   ('VPD', '$T$', 'PAR'), fontsize='large', loc=2)

# plt.savefig('../Fig1/conditions.pdf', bbox_inches='tight')

# plt.setp(host.get_xticklabels(), FontSize=12)
# plt.setp(host.get_yticklabels(), FontSize=12)

# host.axis["left"].label.set_color(p1.get_color())
# par1.axis["right"].label.set_color(p2.get_color())
# par2.axis["right"].label.set_color(p3.get_color())

# plt.draw()
# plt.show()