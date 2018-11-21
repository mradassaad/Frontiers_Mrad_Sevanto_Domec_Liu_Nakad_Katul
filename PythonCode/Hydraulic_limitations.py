import numpy as np
import pandas as pd
from Plant_Env_Props import*
from Useful_Funcs import*
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# -------- Compute hydraulic limitations on Lambda

sample_times = np.array([0, 6/24, 12/24, 14.5/24])
env_data = np.array([cp_interp(sample_times), VPDinterp(sample_times),
                     k1_interp(sample_times), k2_interp(sample_times)])

xvals = np.arange(0.3, 0.5, 0.01)
psi_x_vals = psi_sat * xvals ** -b
psi_l_vals = np.zeros(xvals.shape)
trans_vals = np.zeros(xvals.shape)
i = 0
psi_63 = 3
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    psi_l_vals[i] = OptRes.x
    trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1


def lam(trans, ca, alpha, cp, VPD, k1, k2):
    gl_vals = trans / (VPD * lai)
    part11 = ca ** 2 * gl_vals + 2 * cp * k1 - ca * (k1 - 2 * gl_vals * k2) + k2 * (k1 + gl_vals * k2)
    part12 = np.sqrt(4 * (cp - ca) * gl_vals * k1 + (k1 + gl_vals * (ca + k2)) ** 2)
    part1 = part11 / part12

    part2 = ca + k2

    part3 = 2 * VPD * alpha
    return (part2 - part1) / part3


lam_val_1 = lam(trans_vals, ca, alpha, env_data[0, 1], env_data[1, 1], env_data[2, 1], env_data[3, 1])
lam_val_2 = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])
lam_val_3 = lam(trans_vals, ca, alpha, env_data[0, 3], env_data[1, 3], env_data[2, 3], env_data[3, 3])

psi_63 = 2.5
i = 0
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i],psi_63,w_exp,Kmax))
    psi_l_vals[i] = OptRes.x
    trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1

lam_val_low = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])

psi_63 = 4
i = 0
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    psi_l_vals[i] = OptRes.x
    trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1

lam_val_high = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])

psi_63 = 3
w_exp = 1
i = 0
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    psi_l_vals[i] = OptRes.x
    trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1

lam_val_R = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])

psi_63 = 3
w_exp = 3
i = 0
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    psi_l_vals[i] = OptRes.x
    trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1

lam_val_S = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])

lam_upper_1 = np.ones(lam_val_1.shape) * (ca - env_data[0, 1]) / env_data[1, 1] / alpha
lam_upper_2 = np.ones(lam_val_1.shape) * (ca - env_data[0, 2]) / env_data[1, 2] / alpha
lam_upper_3 = np.ones(lam_val_1.shape) * (ca - env_data[0, 3]) / env_data[1, 3] / alpha

xax = psi_x_vals


fig, ax = plt.subplots()
# plt.plot(xvals, lam_val_1*unit1, 'b-', label='$\psi_{63}=3$, 06:00')
line_ref, = ax.plot(xax, lam_val_2*unit1, 'r-', label='$\psi_{63}=3$, $s=2$')
# plt.plot(xvals, lam_val_3*unit1, 'k-', label='$\psi_{63}=3$, 18:00')
line_low, = ax.plot(xax, lam_val_low*unit1, 'r--', label='$\psi_{63}=2.5$, $s=2$')
line_high, = ax.plot(xax, lam_val_high*unit1, 'r:', label='$\psi_{63}=4$, $s=2$')
line_R, = ax.plot(xax, lam_val_R*unit1, 'b-', label='$\psi_{63}=3$, $s=1$')
line_S, = ax.plot(xax, lam_val_S*unit1, 'k-', label='$\psi_{63}=3$, $s=3$')
# line_upper, = plt.plot(xax, lam_upper_2*unit1, 'g^')
ax.set_xlabel("Soil water potential, $\psi_x$", FontSize=16)
ax.set_ylabel("$\lambda$ lower bound, $\lambda_{lower}$, $mol.m^{-2}$", FontSize=16)
plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)
legend1 = ax.legend((line_ref, line_low, line_high),
                   ('$\psi_{63}=3$', '$\psi_{63}=2.5$', '$\psi_{63}=4$'), fontsize='large', loc=2, title='s=2')
ax.add_artist(legend1)
legend2 = ax.legend((line_ref, line_R, line_S),
                   ('$s=2$', '$s=1$', '$s=3$'), fontsize='large', loc=6, title='$\psi_{63}=3$')
ax.set_xlim(0, 2)
ax.set_ylim(0, 5)
plt.grid(False)



# plt.figure()
# plt.plot(xvals, lam_upper_1*unit1, 'b-', label='06:00')
# plt.plot(xvals, lam_upper_2*unit1, 'r--', label='12:00')
# plt.plot(xvals, lam_upper_3*unit1, 'y:', label='18:00')
# plt.xlabel("Soil Moisture, x")
# plt.ylabel("$\lambda$ upper bound, $\lambda_{upper}$, $mol.m^{-2}$")
# plt.legend()