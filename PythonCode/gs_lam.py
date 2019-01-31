import numpy as np
from Plant_Env_Props import ca, m_w, rho_w, a
import matplotlib.pyplot as plt

# --------------- k1 is per leaf area so gs is per leaf area ----------

def gs_val(lam, ca, alpha, VPD, k1, k2, cp):

    gpart11 = (ca + k2 - 2 * alpha * lam * VPD) *\
              np.sqrt(alpha * lam * VPD * (ca - cp) * (cp + k2) *
              (ca + k2 - alpha * lam * VPD))  # mol/m2/d

    gpart12 = alpha * lam * VPD * ((ca + k2) - alpha * lam * VPD)  # mol/mol

    gpart21 = gpart11 / gpart12  # mol/m2/d

    gpart22 = (ca - (2 * cp + k2))  # mol/m2/d

    gpart3 = gpart21 + gpart22  # mol/m2/d

    gpart4 = (ca + k2)**2  # mol2/mol2

    gs = k1 * gpart3 / gpart4  # mol/m2/d

    return gs

lam = np.linspace(1, 100, 1000)
lai = 1.5
nu = lai * m_w / rho_w  # m3/mol
n = 0.5
z_r = 0.3  # m
alpha = nu * a / (n * z_r)  # m2/mol
VPD = 0.004  # mol/mol
k1 = 1  # mol/m2/d
k2 = 1.8e-4  # mol/mol
cp = 25e-6  # mol/mol
unit = nu * 1e3 / n / z_r  # mol/m2 to mmol/mol

geez_VPD_4 = gs_val(lam, ca, alpha, VPD, k1, k2, cp) * 1e3 /24 /3600

VPD = 0.008
geez_VPD_8 = gs_val(lam, ca, alpha, VPD, k1, k2, cp) * 1e3 /24 /3600

VPD = 0.016
geez_VPD_16 = gs_val(lam, ca, alpha, VPD, k1, k2, cp) * 1e3 /24 /3600
geez_VPD_16[np.isnan(geez_VPD_16)] = 0
geez_VPD_16[np.less(geez_VPD_16, 0)] = 0

VPD = 0.032
geez_VPD_32 = gs_val(lam, ca, alpha, VPD, k1, k2, cp) * 1e3 /24 /3600
geez_VPD_32[np.isnan(geez_VPD_32)] = 0
geez_VPD_32[np.less(geez_VPD_32, 0)] = 0

fig1, ax1 = plt.subplots()
VPD_4, = ax1.plot(lam * unit, geez_VPD_4, 'k')
VPD_8, = ax1.plot(lam * unit, geez_VPD_8, 'k')
VPD_16, = ax1.plot(lam * unit, geez_VPD_16, 'k')
VPD_32, = ax1.plot(lam * unit, geez_VPD_32, 'k')

plt.setp(ax1.get_xticklabels(), FontSize=12)
plt.setp(ax1.get_yticklabels(), FontSize=12)

ax1.set_xlabel("Marginal water use efficiency, $\lambda$, mmol mol$^{-1}$", FontSize=14)
ax1.set_ylabel("Stomatal conductance, $g_s$, mmol m$^{-2}$ s$^{-1}$", FontSize=14)
ax1.set_ylim(0, 50)
ax1.set_xlim(0, 100 * unit)

ax1.text(15 * unit, 16, 'VPD = 32 mmol mol$^{-1}$', rotation=-50)
ax1.text(33 * unit, 15, 'VPD = 16 mmol mol$^{-1}$', rotation=-30)
ax1.text(12.5, 14.5, 'VPD = 8 mmol mol$^{-1}$', rotation=-18)
ax1.text(12, 27, 'VPD = 4 mmol mol$^{-1}$', rotation=-15)
# legend1 = ax3.legend((res_lam, mid_lam, vul_lam),
#                    ('$\psi_{63}=3$', '$\psi_{63}=2.2$', '$\psi_{63}=1.9$'), fontsize='large', loc=2)
# ax3.add_artist(legend1)
# legend2 = ax3.legend((res_lam, res_lam_low),
#                    ('$\lambda (t)$', '$\lambda_{lower}$'), fontsize='large', loc=9)

# fig1.savefig('../Fig6/lam_t_competition_WUS.pdf', bbox_inches='tight')

E_div_a = 0.12  # mmol /m2 /s per leaf area
lam_12_4 = np.interp(E_div_a/4e-3, np.flip(geez_VPD_4, 0), np.flip(lam, 0))
lam_12_8 = np.interp(E_div_a/8e-3, np.flip(geez_VPD_8, 0), np.flip(lam, 0))
lam_12_16 = np.interp(E_div_a/16e-3, np.flip(geez_VPD_16, 0), np.flip(lam, 0))
lam_12_32 = np.interp(E_div_a/32e-3, np.flip(geez_VPD_32, 0), np.flip(lam, 0))

lams_12 = np.array([lam_12_4, lam_12_8, lam_12_16, lam_12_32])
gss_12 = np.array([E_div_a/4e-3, E_div_a/8e-3, E_div_a/16e-3, E_div_a/32e-3])

E_div_a = 0.16  # mmol /m2 /s
lam_16_4 = np.interp(E_div_a/4e-3, np.flip(geez_VPD_4, 0), np.flip(lam, 0))
lam_16_8 = np.interp(E_div_a/8e-3, np.flip(geez_VPD_8, 0), np.flip(lam, 0))
lam_16_16 = np.interp(E_div_a/16e-3, np.flip(geez_VPD_16, 0), np.flip(lam, 0))
lam_16_32 = np.interp(E_div_a/32e-3, np.flip(geez_VPD_32, 0), np.flip(lam, 0))

lams_16 = np.array([lam_16_4, lam_16_8, lam_16_16, lam_16_32])
gss_16 = np.array([E_div_a/4e-3, E_div_a/8e-3, E_div_a/16e-3, E_div_a/32e-3])

E_div_a = 0.2  # mmol /m2 /s
lam_20_4 = np.interp(E_div_a/4e-3, np.flip(geez_VPD_4, 0), np.flip(lam, 0))
lam_20_8 = np.interp(E_div_a/8e-3, np.flip(geez_VPD_8, 0), np.flip(lam, 0))
lam_20_16 = np.interp(E_div_a/16e-3, np.flip(geez_VPD_16, 0), np.flip(lam, 0))
lam_20_32 = np.interp(E_div_a/32e-3, np.flip(geez_VPD_32, 0), np.flip(lam, 0))

lams_20 = np.array([lam_20_4, lam_20_8, lam_20_16, lam_20_32])
gss_20 = np.array([E_div_a/4e-3, E_div_a/8e-3, E_div_a/16e-3, E_div_a/32e-3])

E_div_a = 0.32  # mmol /m2 /s
# lam_20_4 = np.interp(E_div_a/4e-3, np.flip(geez_VPD_4, 0), np.flip(lam, 0))
lam_32_8 = np.interp(E_div_a/8e-3, np.flip(geez_VPD_8, 0), np.flip(lam, 0))
lam_32_16 = np.interp(E_div_a/16e-3, np.flip(geez_VPD_16, 0), np.flip(lam, 0))
lam_32_32 = np.interp(E_div_a/32e-3, np.flip(geez_VPD_32, 0), np.flip(lam, 0))

lams_32 = np.array([lam_32_8, lam_32_16, lam_32_32])
gss_32 = np.array([E_div_a/8e-3, E_div_a/16e-3, E_div_a/32e-3])

supply_limit_12, = plt.plot(lams_12 * unit, gss_12, 'r:')
supply_limit_16, = plt.plot(lams_16 * unit, gss_16, 'r:')
supply_limit_20, = plt.plot(lams_20 * unit, gss_20, 'r:')
supply_limit_32, = plt.plot(lams_32 * unit, gss_32, 'r:')


ax1.text(3.4, 33, 'E = 0.51 mmol m$^{-2}$ s$^{-1}$', rotation=-75, color='red', fontsize=7)
ax1.text(4.8, 41, 'E = 0.32 mmol m$^{-2}$ s$^{-1}$', rotation=-70, color='red', fontsize=8)
ax1.text(6.7, 35, 'E = 0.26 mmol m$^{-2}$ s$^{-1}$', rotation=-65, color='red', fontsize=8)
ax1.text(10.2, 27, 'E = 0.19 mmol m$^{-2}$ s$^{-1}$', rotation=-65, color='red', fontsize=6)

# fig1.savefig('../gs_lam.pdf', bbox_inches='tight')