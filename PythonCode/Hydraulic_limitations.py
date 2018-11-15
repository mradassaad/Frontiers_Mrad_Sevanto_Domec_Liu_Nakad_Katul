import numpy as np
import pandas as pd
from Plant_Env_Props import*
from Useful_Funcs import*
from scipy.optimize import minimize
import matplotlib.pyplot as plt



# -------- Compute hydraulic limitations on Lambda

sample_times = np.array([0,6/24,12/24,18/24])
env_data = np.array([cp_interp(sample_times), VPDinterp(sample_times),
                     k1_interp(sample_times),k2_interp(sample_times)])

xvals = np.arange(0.2,0.35,0.01)
psi_x_vals = psi_sat * xvals ** -b
psi_l_vals = np.zeros(xvals.shape)
trans_vals = np.zeros(xvals.shape)
i = 0
for x in xvals:
    OptRes = minimize(trans_opt,psi_x_vals[i] , args=(psi_x_vals[i],psi_63,w_exp,Kmax))
    psi_l_vals[i] = OptRes.x
    trans_vals[i] = transpiration(OptRes.x,psi_x_vals[i],psi_63,w_exp,Kmax)
    i += 1

def lam(trans, ca, alpha, cp, VPD, k1, k2):
    gl_vals = trans / (VPD * lai)
    part11 = ca ** 2 * gl_vals + 2 * cp * k1 - ca * (k1 - 2 * gl_vals * k2) + k2 * (k1 + gl_vals * k2)
    part12 = np.sqrt(4 * (cp - ca) * gl_vals * k1 + (k1 + gl_vals * (ca + k2)) ** 2)
    part1 = part11 / part12

    part2 = ca + k2

    part3 = 2 * VPD * alpha
    return (part2 - part1) / part3

lam_val_1 = lam(trans_vals, ca, alpha, env_data[0,1], env_data[1,1], env_data[2,1], env_data[3,1])
lam_val_2 = lam(trans_vals, ca, alpha, env_data[0,2], env_data[1,2], env_data[2,2], env_data[3,2])
lam_val_3 = lam(trans_vals, ca, alpha, env_data[0,3], env_data[1,3], env_data[2,3], env_data[3,3])

lam_upper_1 = np.ones(lam_val_1.shape) * (ca - env_data[0,1]) / env_data[1,1] / alpha
lam_upper_2 = np.ones(lam_val_1.shape) * (ca - env_data[0,2]) / env_data[1,2] / alpha
lam_upper_3 = np.ones(lam_val_1.shape) * (ca - env_data[0,3]) / env_data[1,3] / alpha

plt.figure()
plt.plot(xvals,lam_val_1*unit1,'b-', label='06:00')
plt.plot(xvals,lam_val_2*unit1,'r--', label='12:00')
plt.plot(xvals,lam_val_3*unit1,'y:', label='18:00')
plt.xlabel("Soil Moisture, x")
plt.ylabel("$\lambda$ lower bound, $\lambda_{lower}$, $mol.m^{-2}$")
plt.legend()

plt.figure()
plt.plot(xvals,lam_upper_1, 'b-', label='06:00')
plt.plot(xvals,lam_upper_2, 'r--', label='12:00')
plt.plot(xvals,lam_upper_3, 'y:', label='18:00')
plt.xlabel("Soil Moisture, x")
plt.ylabel("$\lambda$ upper bound, $\lambda_{upper}$, $mol.m^{-2}$")
plt.legend()