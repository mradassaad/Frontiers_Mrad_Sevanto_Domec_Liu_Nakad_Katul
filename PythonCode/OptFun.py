import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


# user defined parameter groups
class SoilRoot:
    def __init__(self, ksat, psat, b, n, Zr, RAI):
        self.ksat = ksat  # saturated conductivity, m/s/MPa
        self.psat = psat  # saturated soil water potential, MPa
        self.b = b  # nonlinearity in soil water retention curve
        self.n = n  # soil porosity
        self.Zr = Zr  # rooting depth, m
        self.RAI = RAI  # root area index
        self.sfc = (psat/(-0.03))**(1/b)
        self.sw = (psat/(-3))**(1/b)

class Xylem:
    def __init__(self, gpmax, psi_63, c):
        self.gpmax = gpmax  # maximum xylem conductivity, m/s/MPa
        self.psi_63 = psi_63  # leaf water potential at 50% loss of conductance, MPa
        self.c = c  # nonlinearity of plant vulnerability curve

class Environment:
    def __init__(self, SoilM, RNet, Time, VPD, LAI):
        self.SoilM = SoilM  # soil moisture every 30 minutes
        self.SoilMIni = SoilM(0)  # Soil moisture at start of drydown
        self.SoilMEnd = SoilM(-1)  # Soil moisture at end of drydown
        self.RNet = RNet  # Net radiation every 30 minutes, J/m2/s
        self.Time = Time  # Time
        self.VPD = VPD
        self.LAI = LAI
        self.mean_LAI = np.mean(LAI)



def max_val(k_opt, H_a, H_d, T_l, T_opt):
    """

    :param: k_opt: value of either J_max or V_cmax at Topt in umol/m2/s
    :param: H_a: parameter describing the peaked function that depends on species and growth conditions in kJ/mol
    :param: H_d: another parameter in kJ/mol
    :param T_l: leaf temperature in K
    :param T_opt: optimal temperature in K
    :return:  the value of J_max or V_cmax at T_l in umol/m2/s
    """
    R = 8.314e-3  # kJ/mol/K
    exp_Ha_val = np.exp(H_a * (T_l - T_opt) / (T_l * R * T_opt))
    exp_Hd_val = np.exp(H_d * (T_l - T_opt) / (T_l * R * T_opt))
    return k_opt * (H_d * exp_Ha_val) / (H_d - H_a * (1 - exp_Hd_val))

def J_val(J_max, PAR):
    """

    :param J_max: potential rate of electron transport at a given temperature in umol/m2/s
    :param PAR: photosynthetically active photon flux density in umol/m2/s
    :param theta: curvature parameter of the light response curve
    :param alpha:
    :return: rate of electron transport at a given temperature and PAR in umol/m2/s
    """
    theta = 0.9
    alpha = 0.3 # quantum yield of electron transport in mol electrons / mol photons
    J = (1 / (2 * theta)) * (alpha * PAR + J_max + np.sqrt((alpha * PAR + J_max)**2 - 4 * theta * alpha * PAR * J_max))
    return J


def A(J, Vc_max, Kc, Ko, Oi, ca, cp, g):
    """

    :param a1: Parameter corresponding to either light- or Rubisco-limited photosynthesis
    :param a2: Parameter corresponding to either light- or Rubisco-limited photosynthesis
    :param ca: ambient CO2 mole fraction in the air in mmol/mol
    :param cp: CO2 concentration at which assimilation is zero or compensation point in mmol/mol
    :param g: stomatal conductance in umol/m2/s
    :return: value of A at a particular value of g
    """

    k1 = J/4
    a2 = Kc * (1 + Oi/Ko)
    k2 = (J / 4) * a2 / Vc_max
    gamma = np.sqrt((k2 + ca + k1/g) ** 2 + 4 * k1 * (cp - ca) / g)

    return 0.5 * (k1 + g * (ca + k2) + g * gamma)


def dAdg(k1, k2, ca, cp, g):
    """

    :param a1: Parameter corresponding to either light- or Rubisco-limited photosynthesis
    :param a2: Parameter corresponding to either light- or Rubisco-limited photosynthesis
    :param ca: ambient CO2 mole fraction in the air in mmol/mol
    :param cp: CO2 concentration at which assimilation is zero or compensation point in mmol/mol
    :param g: stomatal conductance in umol/m2/s
    :return: value of the partial derivative of A with respect to g at a particular value of g
    """
    ca *= 1e3
    cp *= 1e3
    gamma = np.sqrt((k2 + ca + k1 / g) ** 2 + 4 * k1 * (cp - ca) / g)

    return (1 / (2 * g * gamma)) * (g * (k2 + ca) ** 2 + k1 * (k2 - ca) + (k2 + ca) * g * gamma)



# def a_light(alpha_p, em, par, cp):
#     """
#
#     :param alpha_p: absorptivity of the leaf for PAR
#     :param em: maximum quantum efficiency in mol/mol
#     :param par: PAR photon flux density incident on the leaf in umol/m2/s
#     :param cp: CO2 concentration at which assimilation is zero or compensation point in mmol/mol
#     :return: a1 in umol/m2/s and a2 in umol/mol
#     """
#     a1 = alpha_p * em * par
#     a2 = 2 * cp * 1e3
#     return a1, a2
#
#
# def a_rubisco(vm, kc, coa, ko):
#     """
#
#     :param vm: the rubisco capacity in umol/m2/s
#     :param kc: Michaelis constant for CO2 in umol/mol
#     :param coa: oxygen mole fraction in the air in mmol/mol
#     :param ko: inhibition constant for O2 in mmol/mol
#     :return: a1 in umol/m2/s and a2 in umol/mol
#     """
#     a1 = vm
#     a2 = kc * (1 + coa / ko)
#     return a1, a2

def vm_calc(vm_25, Tl):
    """

    :param vm_25:  value of the rubisco capacity at leaf temperature Tl of 25C in umol/m2/s
    :param Tl: leaf temperature in C
    :return: the rubisco capacity Vm in umol/m2/s
    """
    # The optimal temperature is considered 25C
    # The cutoff temperature is assumed 41C
    # For reference, check Norman and Campbell page 239-241
    return vm_25 * np.exp(0.088 * (Tl - 25)) / (1 + np.exp(0.29 * (Tl - 41)))

def dydt(t, y):
    """
    y[0] is lambda(t) and y[1] is x(t)
    t is the time
    """

    t_day = 0.7  # day/day
    unit0 = 24 * 3600 * t_day  # 1/s -> 1/d

    gamma = 0.01  # m/day
    vpd = 0.015  # mol/mol
    k = 0.05 * unit0  # mol/m2/day
    z_r = 0.3  # m
    ca = 350e-6  # mol/mol
    a = 1.6
    n = 0.5  # m3/m3
    lai = 1.5
    m_w = 0.018  # kg/mol

    rhow = 1000  # kg/m3
    nu = lai * m_w * (t_day) / rhow  # m3/mol
    c = 1
    beta = gamma / (n * z_r)  # 1/day
    alpha = nu * a / (n * z_r)  # m2/mol

    unit1 = 10 ** 3 * nu / (n * z_r)  # mol/m2 -> mmol/mol
    unit2 = 18 * 1e-6  # mol H2O/m2/s ->  m/s
    unit3 = 1e6  # Pa -> MPa
    unit4 = 273.15  # Degree C -> K


    g = k * (np.sqrt(ca/(alpha*vpd*y[0]))-1)  # mol/m2/day

    losses = beta*y[1]**c  # 1/day
    evap_trans = alpha*g*vpd  # 1/day
    f = -(losses + evap_trans)  # 1/day

    dlamdt = y[0]*beta*c*y[1]**(c-1)  #mol/m2/day
    dxdt = f  # 1/day

    return np.vstack((dlamdt, dxdt))


def Weibull(Xylem,psi_s,psi_l):

    return Xylem.gpmax * (np.exp( ((psi_s - psi_l) / Xylem.psi_63) ** Xylem.c ))


def bc(ya, yb):  # boundary imposed on x at t=T
    x0 = 0.8
    return np.array([ya[1] - x0, yb[1]])


def bc_wus(ya,yb):  # Water use strategy
    x0 = 0.8
    wus_coeff = 800e-6*t_day*unit0  # mol/m2
    return np.array([ya[1] - x0, yb[0] - wus_coeff])


t = np.linspace(0, 20, 1000)

lam_guess = 1*np.ones((1, t.size))
x_guess = 0.8*np.ones((1, t.size))

y_guess = np.vstack((lam_guess, x_guess))

res = solve_bvp(dydt, bc, t, y_guess)

lam_plot = res.sol(t)[0]*unit1
soilM_plot = res.sol(t)[1]
plt.subplot(311)
plt.plot(t, lam_plot)
#plt.xlabel("days")
plt.ylabel("$\lambda (t), mmol.mol^{-1}$")

plt.subplot(312)
plt.plot(t, (k/unit0) * (np.sqrt(ca/(alpha*vpd*res.sol(t)[0]))-1))
plt.ylabel("$g(t), mol.m^{-2}.s^{-1}$")

plt.subplot(313)
plt.plot(t, soilM_plot)
plt.xlabel("time, days")
plt.ylabel("$x(t)$")

plt.show()