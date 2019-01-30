import numpy as np
import pandas as pd
from Plant_Env_Props import*
from Useful_Funcs import*
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --------- Maximum transpiration rate and leaf water potential vs soil water potential ------

x = np.arange(0.1, 0.25, 0.005)

psi_sat = 1.5 * unit6  # Soil water potential at saturation, MPa
b = 3.1  # other parameter
RAI = 5  # m3 m-3
gamma = 0.00072 * unit5   # m/d, for sandy loam page 130 Campbell and Norman
c = 2*b+3

psi_63 = 3  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 2  # Weibull exponent
Kmax = 2e-3 * unit0  # Maximum plant stem water leaf area-averaged conductivity, mol/m2/d/MPa
reversible = 1

psi_x = psi_sat * x ** -b
psi_l = np.zeros(x.shape)
psi_r = np.zeros(x.shape)
trans_max = np.zeros(x.shape)
i = 0
for xx in x:
    OptRes = minimize(trans_opt, psi_x[i], args=(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible))
    psi_l[i] = OptRes.x
    trans_res = transpiration(OptRes.x, x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible)
    trans_max[i] = trans_res[0]
    psi_r[i] = trans_res[1]
    i += 1

trans_max_interp = interp1d(psi_x, trans_max, kind='cubic')
psi_l_interp = interp1d(psi_x, psi_l, kind='cubic')
psi_r_interp = interp1d(psi_x, psi_r, kind='cubic')

gSR = gSR_val(x, gamma, b, d_r, z_r, RAI)
gRL = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)

fig, ax = plt.subplots()
ax.plot(x, gSR * 1e3 / unit0)
ax.plot(x, gRL * 1e3 / unit0)
ax.set_ylim(0, 2)

ax.plot(x, trans_max_interp(psi_x) * 1e3 / unit0)

# -------- Compute hydraulic limitations on Lambda
#
# sample_times = np.array([6/24, 12/24, 14.5/24, 18/24])
# env_data = np.array([cp_interp(sample_times), VPDinterp(sample_times),
#                      k1_interp(sample_times), k2_interp(sample_times)])
#
# xvals = np.arange(0.12, 0.5, 0.001)
# psi_x_vals = psi_sat * xvals ** -b
# psi_l_vals = np.zeros(xvals.shape)
# psi_r_vals = np.zeros(xvals.shape)
# trans_vals = np.zeros(xvals.shape)
# i = 0
# psi_63 = 3
# w_exp = 2
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI))
#     psi_l_vals[i] = OptRes.x
#     trans_res = transpiration(OptRes.x, xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI)
#     trans_vals[i] = trans_res[0]
#     psi_r_vals[i] = trans_res[1]
#     i += 1
#
#
# def lam(trans, ca, alpha, cp, VPD, k1, k2):
#     gl_vals = trans / (VPD * lai)
#     part11 = ca ** 2 * gl_vals + 2 * cp * k1 - ca * (k1 - 2 * gl_vals * k2) + k2 * (k1 + gl_vals * k2)
#     part12 = np.sqrt(4 * (cp - ca) * gl_vals * k1 + (k1 + gl_vals * (ca + k2)) ** 2)
#     part1 = part11 / part12
#
#     part2 = ca + k2
#
#     part3 = 2 * VPD * alpha
#     return (part2 - part1) / part3
#
#
# lam_val_0 = lam(trans_vals, ca, alpha, env_data[0, 0], env_data[1, 0], env_data[2, 0], env_data[3, 0])
# lam_val_1 = lam(trans_vals, ca, alpha, env_data[0, 1], env_data[1, 1], env_data[2, 1], env_data[3, 1])
# lam_val_2 = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])
# lam_val_3 = lam(trans_vals, ca, alpha, env_data[0, 3], env_data[1, 3], env_data[2, 3], env_data[3, 3])
#
# lam_upper_0 = np.ones(lam_val_1.shape) * (ca - env_data[0, 0]) / env_data[1, 0] / alpha
# lam_upper_1 = np.ones(lam_val_1.shape) * (ca - env_data[0, 1]) / env_data[1, 1] / alpha
# lam_upper_2 = np.ones(lam_val_1.shape) * (ca - env_data[0, 2]) / env_data[1, 2] / alpha
# lam_upper_3 = np.ones(lam_val_1.shape) * (ca - env_data[0, 3]) / env_data[1, 3] / alpha

# fig, ax = plt.subplots()
# line_low_0, = plt.plot(-psi_x_vals, lam_val_0 * unit1, 'r')
# line_low_1, = plt.plot(-psi_x_vals, lam_val_1 * unit1, 'k')
# # plt.plot(psi_x_vals, lam_val_2 * unit1)
# line_low_3, = plt.plot(-psi_x_vals, lam_val_3 * unit1, 'b')
# line_high_0, = plt.plot(-psi_x_vals, lam_upper_0 * unit1, 'r--')
# line_high_1, = plt.plot(-psi_x_vals, lam_upper_1 * unit1, 'k--')
# line_high_3, = plt.plot(-psi_x_vals, lam_upper_3 * unit1, 'b--')
# ax.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=16)
# ax.set_ylabel("$\lambda$ lower and upper bounds, mol.m$^{-2}$", FontSize=16)
# plt.setp(ax.get_xticklabels(), FontSize=12)
# plt.setp(ax.get_yticklabels(), FontSize=12)
# legend1 = ax.legend((line_low_0, line_low_1, line_low_3),
#                    ('06:00', '12:00', '18:00'), fontsize='large', loc=3, title='Color')
# ax.add_artist(legend1)
# legend2 = ax.legend((line_low_1, line_high_1),
#                    ('$\lambda_{lower}$', '$\lambda_{upper}$'), fontsize='large', loc=8, title='Line Style')

# plt.savefig('limits_time.pdf', bbox_inches='tight')


# psi_63 = 2
# i = 0
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI))
#     psi_l_vals[i] = OptRes.x
#     trans_res = transpiration(OptRes.x, xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI)
#     trans_vals[i] = trans_res[0]
#     psi_r_vals[i] = trans_res[1]
#     i += 1
#
# lam_val_low = lam(trans_vals, ca, alpha, env_data[0, 1], env_data[1, 1], env_data[2, 1], env_data[3, 1])
#
# psi_63 = 3
# w_exp = 1
# i = 0
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI))
#     psi_l_vals[i] = OptRes.x
#     trans_res = transpiration(OptRes.x, xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI)
#     trans_vals[i] = trans_res[0]
#     psi_r_vals[i] = trans_res[1]
#     i += 1
#
# lam_val_high = lam(trans_vals, ca, alpha, env_data[0, 1], env_data[1, 1], env_data[2, 1], env_data[3, 1])
# #
# psi_63 = 3
# w_exp = 1
# i = 0
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
#     psi_l_vals[i] = OptRes.x
#     trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
#     i += 1
#
# lam_val_R = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])
#
# psi_63 = 3
# w_exp = 3
# i = 0
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
#     psi_l_vals[i] = OptRes.x
#     trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
#     i += 1
#
# lam_val_S = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])
#
# lam_upper_1 = np.ones(lam_val_1.shape) * (ca - env_data[0, 1]) / env_data[1, 1] / alpha
# lam_upper_2 = np.ones(lam_val_1.shape) * (ca - env_data[0, 2]) / env_data[1, 2] / alpha
# lam_upper_3 = np.ones(lam_val_1.shape) * (ca - env_data[0, 3]) / env_data[1, 3] / alpha
#
# xax = psi_x_vals
#
#
# fig, ax = plt.subplots()
# # plt.plot(xvals, lam_val_1*unit1, 'b-', label='$\psi_{63}=3$, 06:00')
# line_ref, = ax.plot(xax, lam_val_2*unit1, 'r-', label='$\psi_{63}=3$, $s=2$')
# # plt.plot(xvals, lam_val_3*unit1, 'k-', label='$\psi_{63}=3$, 18:00')
# line_low, = ax.plot(xax, lam_val_low*unit1, 'r--', label='$\psi_{63}=2.5$, $s=2$')
# line_high, = ax.plot(xax, lam_val_high*unit1, 'r:', label='$\psi_{63}=4$, $s=2$')
# line_R, = ax.plot(xax, lam_val_R*unit1, 'b-', label='$\psi_{63}=3$, $s=1$')
# line_S, = ax.plot(xax, lam_val_S*unit1, 'k-', label='$\psi_{63}=3$, $s=3$')
# # line_upper, = plt.plot(xax, lam_upper_2*unit1, 'g^')
# ax.set_xlabel("Soil water potential, $\psi_x$", FontSize=16)
# ax.set_ylabel("$\lambda$ lower bound, $\lambda_{lower}$, $mol.m^{-2}$", FontSize=16)
# plt.setp(ax.get_xticklabels(), FontSize=12)
# plt.setp(ax.get_yticklabels(), FontSize=12)
# legend1 = ax.legend((line_ref, line_low, line_high),
#                    ('$\psi_{63}=3$', '$\psi_{63}=2.5$', '$\psi_{63}=4$'), fontsize='large', loc=2, title='s=2')
# ax.add_artist(legend1)
# legend2 = ax.legend((line_ref, line_R, line_S),
#                    ('$s=2$', '$s=1$', '$s=3$'), fontsize='large', loc=6, title='$\psi_{63}=3$')
# ax.set_xlim(0, 2)
# ax.set_ylim(0, 5)
# plt.grid(False)



# plt.figure()
# plt.plot(xvals, lam_upper_1*unit1, 'b-', label='06:00')
# plt.plot(xvals, lam_upper_2*unit1, 'r--', label='12:00')
# plt.plot(xvals, lam_upper_3*unit1, 'y:', label='18:00')
# plt.xlabel("Soil Moisture, x")
# plt.ylabel("$\lambda$ upper bound, $\lambda_{upper}$, $mol.m^{-2}$")
# plt.legend()