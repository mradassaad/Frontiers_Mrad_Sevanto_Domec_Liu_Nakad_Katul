import numpy as np
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.special import gammaincc
from scipy.special import gamma as gammafunc
from scipy.special import gammainccinv

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class GuessError(Error):
    """Exception raised for lack of satisfactory BVP solution due
     to possible wrongful inital guess for lambda.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

def max_val(k_opt, H_a, H_d, Tl, T_opt):
    """

    :param: k_opt: value of either J_max or V_cmax at Topt in umol/m2/s per unit LEAF area
    :param: H_a: parameter describing the peaked function that depends on species and growth conditions in kJ/mol
    :param: H_d: another parameter in kJ/mol
    :param Tl: leaf temperature in K
    :param T_opt: optimal temperature in K
    :return:  the value of J_max or V_cmax at T_l in umol/m2/s
    """
    R = 8.314e-3  # kJ/mol/K
    exp_Ha_val = np.exp(H_a * (Tl - T_opt) / (Tl * R * T_opt))
    exp_Hd_val = np.exp(H_d * (Tl - T_opt) / (Tl * R * T_opt))
    return k_opt * (H_d * exp_Ha_val) / (H_d - H_a * (1 - exp_Hd_val))


def max_val_j_25(k_25, Tl):

    R = 8.314  # J/mol/K
    part_1 = 1 + np.exp((-4534)/ (298.2 * R))
    part_2 = np.exp((50300 / (298.2 * R)) * (1 - 298.2 / Tl))
    part_3 = 1 + np.exp((495 * Tl - 152044) / (R * Tl))
    return k_25 * part_1 * part_2 / part_3

def max_val_v_25(k_25, Tl):

    R = 8.314  # J/mol/K
    part_1 = 1 + np.exp(-4424 / (298.2 * R))
    part_2 = np.exp((73637 / (298.2 * R)) * (1 - 298.2 / Tl))
    part_3 = 1 + np.exp((486 * Tl - 149252) / (R * Tl))
    return k_25 * part_1 * part_2 / part_3


def RNtoPAR(RN):
    """

    :param RN: radiation in W/m2
    :return: PAR in umol/m2/s
    """
    hc = 2e-25  # Planck constant times light speed, J*s times m/s
    wavelen = 500e-9  # wavelength of light, m
    EE = hc / wavelen  # energy of photon, J
    NA = 6.02e23  # Avogadro's constant, /mol
    PAR = RN / (EE * NA) * 1e6  # absorbed photon irradiance, umol photons /m2/s, PAR per unit GROUND area
    return PAR


def J_val(PAR, Jmax):
    """

    :param PAR: photosynthetically active photon flux density in umol/m2/s - per LEAF area
    :return: rate of electron transport at a given temperature and PAR in umol/m2/s
    """

    # J_max: potential rate of electron transport at a given temperature in umol/m2/s
    theta = 0.9  # curvature parameter of the light response curve
    alpha = 0.3 # quantum yield of electron transport in mol electrons / mol photons
    J = (1 / (2 * theta)) * \
        (alpha * PAR + Jmax - np.sqrt((alpha * PAR + Jmax)**2 -
                                      4 * theta * alpha * PAR * Jmax))
    return J


def MMcoeff(Tl):
    """

    :param Tl: leaf temperature in K
    :return: Kc and Ko are the Michaelis-Menten coefficients for Rubisco for CO2 and O2
    """
    # Bernacchi et al. 2001 (in vivo)
    R = 8.314  # J/mol/K
    Kc = 404.9 * np.exp(79430 * (Tl - 298) / (298 * R * Tl))  # umol/mol
    Ko = 278.4 * np.exp(36380 * (Tl - 298) / (298 * R * Tl))  # mmol/mol
    return Kc, Ko


def cp_val(Tl):
    """

    :param Tl: leaf temperature in K
    :return: cp is the compensation point for CO2 in umol/mol
    """

    R = 8.314  # J/mol/K
    cp = 42.75 * np.exp(37830 * (Tl - 298) / (298 * R * Tl))  # umol/mol
    return cp


def A(t, gs, ca, k1_interp, k2_interp, cp_interp, phi=1):
    """

    :param gs: stomatal conductance in umol/m2/s per unit LEAF area
    :return: value of A at a particular value of g in mol/m2/s
    """
    '''J: Electron transport rate in umol/m2/s
     Vc_max: maximum rate of rubisco activity in umol/m2/s
    Kc: Michaelis-Menten constant for CO2 in umol/mol
     Ko: Michaelis-Menten constant for O2 in mmol/mol
     ca: ambient CO2 mole fraction in the air in umol/mol
    cp: CO2 concentration at which assimilation is zero or compensation point in umol/mol'''

    B = (cp_interp(t) + k2_interp(t)) / phi  # non-stomatal limitation; mesophyll

    delta = np.sqrt((B + ca - cp_interp(t)) ** 2 * gs ** 2 +
                    2 * gs * k1_interp(t) * (B - ca + cp_interp(t)) + k1_interp(t) ** 2)  # mol/m2/d

    A = 0.5 * ((B + ca - cp_interp(t)) * gs + k1_interp(t) - delta)  # mol/m2/d
    # A *= 1e6/unit0

    return A


# def soil_water_pot(x, b, psi_sat):
#
#     return psi_sat * x ** (-b)


def Interstorm(df, drydownid):
    nobsinaday = 48  # number of observations in a day

    dailyP = dailyAvg(np.array(df['P']), nobsinaday).ravel()
    rainyday = np.where(dailyP > 0)[0]
    drydownlength = np.concatenate([np.diff(rainyday), [0]])
    id1 = rainyday[drydownlength > 30]+1  # start day of each dry down period longer than 30 days
    id2 = id1+drydownlength[drydownlength > 30]-1  # end day of each dry down period
    st = list(df['TIMESTAMP_START'][id1*nobsinaday-1])
    et = list(df['TIMESTAMP_START'][id2*nobsinaday-1])
#     print([st,et])
    print('Selected period: '+str(st[drydownid])+' to '+str(et[drydownid]))
   
    import pandas as pd
    from datetime import datetime
#     import pdb; pdb.set_trace()
    
    return df[(pd.to_datetime(df['TIMESTAMP_START'], format='%m/%d/%Y %H:%M') >= datetime.strptime(st[drydownid], '%m/%d/%Y %H:%M')) &  (pd.to_datetime(df['TIMESTAMP_START'], format='%m/%d/%Y %H:%M') < datetime.strptime(et[drydownid], '%m/%d/%Y %H:%M'))]


def dailyAvg(data, windowsize):
    data = np.array(data)
    data = data[0:windowsize*int(len(data)/windowsize)]
    return np.nanmean(np.reshape(data,
                            [int(len(data)/windowsize), windowsize]), axis=1)


# Anet = A(J, Vmax, Kc, Ko, Oi, 350, cp, 0.1e6)

# plt.figure()
# plt.plot(Anet, '-k')
# plt.xlim([0, 48*5])
# plt.xlabel('Time step (half-hour)')
# plt.ylabel(r'An ($\mu$mol CO$_2$ /m$^2$/s)')

def g_val(t, lam, ca, VPDinterp, k1_interp, k2_interp, cp_interp):

    a = 1.6

    # --------------- k1 is per leaf area so gs is per leaf area ----------

    gpart11 = (ca + k2_interp(t) - 2 * a * lam * VPDinterp(t)) *\
              np.sqrt(a * lam * VPDinterp(t) * (ca - cp_interp(t)) * (cp_interp(t) + k2_interp(t)) *
              (ca + k2_interp(t) - a * lam * VPDinterp(t)))  # mol/mol

    gpart12 = a * lam * VPDinterp(t) * ((ca + k2_interp(t)) - a * lam * VPDinterp(t))  # mol/mol

    gpart21 = gpart11 / gpart12  # mol/mol

    gpart22 = ca - 2 * cp_interp(t) - k2_interp(t)  # mol/mol

    gpart3 = gpart21 + gpart22  # mol/mol

    gpart4 = (ca + k2_interp(t))**2  # mol2 mol-2

    gs = k1_interp(t) * gpart3 / gpart4  # mol/m2/d per unit LEAF area


    # B = (cp_interp(t) + k2_interp(t)) / phi  # non-stomatal limitation; mesophyll
    #
    # gpart11 = (B - cp_interp(t) + ca - 2 * a * VPDinterp(t) * lam) * \
    #           np.sqrt(a * VPDinterp(t) * lam * B * k1_interp(t) ** 2 * (ca - cp_interp(t)) *
    #                   (B - cp_interp(t) + ca - a * VPDinterp(t) * lam))
    #
    # gpart12 = a * VPDinterp(t) * lam * (B - cp_interp(t) + ca) ** 2 *\
    #           (B - cp_interp(t) + ca - a * VPDinterp(t) * lam)
    #
    # gpart21 = k1_interp(t) * (B - ca + cp_interp(t))
    #
    # gpart22 = (B + ca - cp_interp(t)) ** 2
    #
    # gs = - gpart21 / gpart22 + gpart11 / gpart12

    return gs


def create_ranges(start, stop, N, endpoint=True):
    if endpoint:
        divisor = N-1
    else:
        divisor = N
    steps = (1.0/divisor) * (stop - start)
    return steps[:, None]*np.arange(N) + start[:, None]


def psil_val_integral(psi_l, psi_r, psi_63, w_exp, Kmax, gs, VPDinterp, t, reversible=0):

    trans_res = plant_cond_integral(psi_l, psi_r, psi_63, w_exp, Kmax, reversible)

    if np.any(
            np.logical_not(np.isfinite(trans_res - 1.6 * gs * VPDinterp(t)))):
        raise GuessError('Try increasing the lambda guess or there may be no solutions for the parameter choices.')

    return trans_res - 1.6 * gs * VPDinterp(t)  # all in unit LEAF area


# def psil_val(psi_l, psi_r, psi_63, w_exp, Kmax, gs, VPDinterp, t, reversible=0):
#
#     trans_res = (psi_l - psi_r) * plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
#
#     if np.any(
#             np.logical_not(np.isfinite(trans_res - 1.6 * gs * VPDinterp(t)))):
#         raise GuessError('Try increasing the lambda guess or there may be no solutions for the parameter choices.')
#
#     return trans_res - 1.6 * gs * VPDinterp(t)  # all in unit LEAF area


def psi_r_val(x, psi_sat, gamma, b, d_r, z_r, RAI, gl, lai, VPDinterp, t):

    psi_x = psi_sat * x ** -b
    gSR = gSR_val(x, gamma, b, d_r, z_r, RAI, lai)  # per unit LEAF area
    trans = 1.6 * gl * VPDinterp(t)  # per unit LEAF area
    return psi_x + trans / gSR



def transpiration(psi_l, x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, reversible=0):
    """

    :param psi_l: leaf water potential in MPa
    :param x: Soil moisture in m3/m3
    :param psi_sat: Soil saturation pressure in MPa
    :param gamma: Saturated soil hydraulic conductivity in m/d
    :param b: Soil retention curve exponent
    :param psi_63: Weibull parameter in MPa
    :param w_exp: Weibull exponent
    :param Kmax: leaf area-averaged plant hydraulic conductivity in mol/m2/MPa/d
    :param d_r: diameter of fine roots in meters
    :param z_r: rooting depth in meters
    :param RAI: Root Area Index in m/m
    :return: transpiration rate in mol/m2/d per unit LEAF area
    """
    psi_x = psi_sat * x ** -b
    # if np.less(np.abs(psi_x - psi_l), 1e-5):
    #     return 0, psi_x

    gSR = gSR_val(x, gamma, b, d_r, z_r, RAI, lai)
    plant_x = grl_val((psi_x + psi_l) / 2, psi_63, w_exp, Kmax)
    psi_r_guess = (psi_x * gSR + psi_l * plant_x) / (gSR + plant_x)
    # psi_r_guess = psi_x + 0.1
    #
    # res = root(lambda psi_r: (psi_r - psi_x) * gSR -
    #                         (psi_l - psi_r) * plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible),
    #            psi_r_guess, method='hybr')

    res = root(lambda psi_r: (psi_r - psi_x) * gSR - plant_cond_integral(psi_l, psi_r, psi_63, w_exp, Kmax, 1),
               psi_r_guess, method='hybr')

    psi_r = res.get('x')

    #  returns transpiration in mol/m2/d per unit LEAF area
    return (psi_r - psi_x) * gSR, psi_r


def trans_crit(x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, endP=10):
    #  per unit LEAF area
    psi_x = psi_sat * x ** (-b)

    PP = np.arange(psi_x, endP, 0.01)
    EE, psi_r = transpiration(PP, x, psi_sat, gamma, b,
                          psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, 1)
    kk = np.gradient(EE, PP, edge_order=2)
    k_max = kk[0]
    # Find critical point

    crit_ind = np.argmin(np.abs(kk / k_max - 0.05))

    E_crit = EE[crit_ind]  # mol m-2 d-1; per unit leaf area
    k_crit = kk[crit_ind]
    P_crit = PP[crit_ind]

    # _, Pcrit, k_crit = rel_loss(psi_x, psi_x + 0.1, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai)
    # trans_res, psi_r = transpiration(Pcrit, x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, reversible)

    return E_crit, psi_r[crit_ind], P_crit, k_crit, k_max


def trans_opt(psi_l, x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, reversible=0):
    #  per unit LEAF area
    return - transpiration(psi_l, x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, reversible)[0]


# def trans_max_val(x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, reversible=0):
#     #  per unit LEAF area
#     psi_x = psi_sat * x ** -b
#     OptRes = minimize(trans_opt, psi_x,
#                       args=(x,
#                             psi_sat, gamma, b, psi_63, w_exp, Kmax,
#                             d_r, z_r, RAI, lai, reversible))
#     psi_l_max = OptRes.x
#     trans_res = transpiration(OptRes.x, x,
#                               psi_sat, gamma, b, psi_63, w_exp, Kmax,
#                               d_r, z_r, RAI, lai, reversible)
#     trans_max = trans_res[0]
#     psi_r_max = trans_res[1]
#
#     return trans_max, psi_l_max, psi_r_max


# def dtransdx(psi_l, x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible=0):
#
#     psi_x = psi_sat * x ** -b
#     dpsi_xdx = -b * psi_sat * x ** (-b-1)
#     gSR = gSR_val(x, gamma, b, d_r, z_r, RAI)
#     dgSR_dx = dgSR_dx_val(x, gamma, b, d_r, z_r, RAI)
#     psi_r = transpiration(psi_l, x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible)[1]
#     ktot = (plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible) ** -1 + gSR_val(x, gamma, b, d_r, z_r, RAI) ** -1) ** -1
#     dEdx = - dpsi_xdx * ktot + (psi_l - psi_x) * dgSR_dx * ktot ** 2 / (gSR ** 2)
#
#     return dEdx  # mol/m2/d

def plant_cond_integral(psi_l, psi_r, psi_63, w_exp, Kmax, reversible=0):
    """

    :param psi_r: root water potential in MPa
    :param psi_l: leaf water potential in MPa
    :param psi_63: Weibull parameter in MPa
    :param w_exp: Weibull exponent
    :param Kmax: Saturated plant LEAF area-average conductance in mol/m2/MPa/d
    :return: Unsaturated plant LEAF area-average conductance in mol/m2/MPa/d
    """
    cond_pot = gammafunc(1 / w_exp) * Kmax * psi_63 / w_exp *\
               (gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp) -
                gammaincc(1 / w_exp, (psi_l / psi_63) ** w_exp))


    if reversible:
        return cond_pot
    else:

        cond_pot = np.minimum.accumulate(cond_pot)

        return cond_pot

def grl_val(psi, psi_63, w_exp, Kmax):
    """

    :param psi_r: root water potential in MPa
    :param psi_l: leaf water potential in MPa
    :param psi_63: Weibull parameter in MPa
    :param w_exp: Weibull exponent
    :param Kmax: Saturated plant LEAF area-average conductance in mol/m2/MPa/d
    :return: Unsaturated plant LEAF area-average conductance in mol/m2/MPa/d
    """
    return Kmax * np.exp(- (psi / psi_63) ** w_exp)


def gSR_val(x, gamma, b, d_r, z_r, RAI, lai):
    """

    :param x: Soil moisture in m3.m-3
    :param gamma: Saturated hydraulic conductivity of soil in m.d-1
    :param b: exponent of relation
    :param d_r: diameter of fine roots in meters
    :param z_r: rooting depth in meters
    :param RAI: Root Area Index in m/m
    :return: soil to root hydraulic conductance in mol/m2/MPa/d per LEAF area
    """
    lSR = np.sqrt(d_r * z_r / RAI)
    ks = gamma * x ** (2*b+3)  # Unsaturated hydraulic conductivity of soil in m/d
    unit = 24 * 3600  # 1/s -> 1/d
    grav = 9.81  # gravitational acceleration in N/Kg
    M_w = 0.018  # molar mass of water in Kg/mol
    gSR = ks / unit / grav / lSR  # kg/N/s
    gSR *= 1e6 / M_w  # mol/MPa/m2/s
    gSR *= unit / lai  # mol/MPa/m2/d per unit LEAF area
    return gSR


def lam_from_trans(trans, ca, cp, VPD, k1, k2):
    gs_vals = trans / (1.6 * VPD)  # per unit LEAF area
    part11 = ca ** 2 * gs_vals + 2 * cp * k1 - ca * (k1 - 2 * gs_vals * k2) + k2 * (k1 + gs_vals * k2)
    part12 = np.sqrt(4 * (cp - ca) * gs_vals * k1 + (k1 + gs_vals * (ca + k2)) ** 2)
    part1 = part11 / part12

    part2 = ca + k2

    part3 = 2 * VPD * 1.6
    return (part2 - part1) / part3  # mol/mol


# def rel_loss(psi_x, psi_l, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai):
#
#     # k_x_max = np.exp(-(psi_x / psi_63) ** w_exp)
#     # k_l_max = np.exp(-(psi_l / psi_63) ** w_exp)
#     # k_crit = 0.05 * k_x_max
#     # Pcrit = psi_63 * (- np.log(k_crit)) ** (1 / w_exp)
#     rhizo_cond = gSR_val((psi_x/psi_sat)**(-1/b), gamma, b, d_r, z_r, RAI, lai)
#
#     k_x_max = (1/plant_cond(psi_x, psi_x, psi_63, w_exp, Kmax, 1) + 1/rhizo_cond) ** (-1)
#
#     k_l_max = (1 / plant_cond(psi_l, psi_l, psi_63, w_exp, Kmax, 1) + 1 / rhizo_cond) ** (-1)
#
#     k_crit = 0.05 * k_x_max
#
#     Pcrit = psi_63 * (- np.log((-1 / rhizo_cond + 1/k_crit) ** (-1) / Kmax)) ** (1 / w_exp)
#
#     return (k_x_max - k_l_max) / (k_x_max - k_crit), Pcrit, k_crit


def A_here(gl, ca, k1, k2, cp):
    delta = np.sqrt(((k2 + ca) * gl + k1) ** 2 -
                        4 * k1 * (ca - cp) * gl)  # mol/mol

    AAA = 0.5 * (k1 + gl * (ca + k2) - delta)  # mol/m2/s
    # A *= 1e6/unit0

    return AAA


def profit_max(psi_x, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, ca,
               k1, k2, cp, VPD, step=0.02, crits=np.array((0, 0, 0))):

    if np.less(k1, 100 * np.finfo(float).eps):
        return 0, 0, psi_x, 0, 0, psi_x, psi_x
    x = (psi_x / psi_sat) ** (-1 / b)
    # _, Pcrit, _ = rel_loss(psi_x, psi_x + 0.1, psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai)

    gSR = gSR_val(x, gamma, b, d_r, z_r, RAI, lai)  # per unit LEAF area

    if crits[0] == 0:
        PP = np.arange(psi_x, 7, 0.1)
        EE, _ = transpiration(PP, x, psi_sat, gamma, b,
                    psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, 1)
        kk = np.gradient(EE, PP)
        k_max = kk[0]
    # Find critical point

        crit_ind = np.argmin(np.abs(kk / k_max - 0.05))

        E_crit = EE[crit_ind]  # mol m-2 d-1; per unit leaf area
        k_crit = kk[crit_ind]
        P_crit = PP[crit_ind]
    else:
        E_crit = crits[1]  # mol m-2 d-1; per unit leaf area
        k_crit = crits[2]
        k_max = crits[3]
        P_crit = crits[0]

    # Finer Mesh

    PP = np.arange(psi_x, P_crit, step)
    EE, psirr = transpiration(PP, x, psi_sat, gamma, b,
                          psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, 1)
    kk = grl_val(PP, psi_63, w_exp, Kmax) * gSR_val(x, gamma, b, d_r, z_r, RAI, lai) / \
         (grl_val(psirr, psi_63, w_exp, Kmax) + gSR_val(x, gamma, b, d_r, z_r, RAI, lai))

    ggss = EE / 1.6 / VPD  # mol/m2/d

    AA = A_here(ggss, ca, k1, k2, cp)

    gs_crit = E_crit / 1.6 / VPD  # mol m-2 d-1
    A_crit = A_here(gs_crit, ca, k1, k2, cp)

    trans = 1.6 * ggss * VPD  # per unit LEAF area
    psi_r = psi_x + trans / gSR

    rel_gain = np.gradient(AA / A_crit, PP)
    rel_loss = np.gradient((k_max - kk) / (k_max - k_crit), PP)

    opt_ind = np.argmin(np.abs(rel_gain - rel_loss))

    P_opt = PP[opt_ind]
    psi_r_opt = psi_r[opt_ind]
    A_opt = AA[opt_ind]
    E_opt = EE[opt_ind]

    return E_crit, A_crit, P_crit, E_opt, A_opt, P_opt, psi_r_opt

# ------------------------- Methods for mesophyll -------------------------


def phi_val(psi_l, psi_crit):

    return 1 - psi_l / psi_crit


def dphi_dgs(psi_l, psi_crit, psi_x, psi_r, VPD, psi_63, w_exp, Kmax, psi_sat,
             gamma, b, d_r, z_r, RAI, lai):
    x = (psi_x / psi_sat) ** - (1 / b)
    a = 1.6
    part1 = a * VPD / grl_val(psi_l, psi_63, w_exp, Kmax)
    part2 = 1 + grl_val(psi_r, psi_63, w_exp, Kmax) / gSR_val(x, gamma, b, d_r, z_r, RAI, lai)
    return - part1 * part2 / psi_crit


def dpsix_dx(x, psi_sat, b):

    return - psi_sat * b * x ** (-b - 1)


def dgSR_dx(x, gamma, b, d_r, z_r, RAI, lai):
    """

    :param x: Soil moisture in m3.m-3
    :param gamma: Saturated hydraulic conductivity of soil in m.d-1
    :param b: exponent of relation
    :param d_r: diameter of fine roots in meters
    :param z_r: rooting depth in meters
    :param RAI: Root Area Index in m/m
    :return: partial derivative of the soil to root hydraulic conductivity wrt x in mol/m2/MPa/d
    """
    lSR = np.sqrt(d_r * z_r / RAI)
    dgx_dx = (2*b+3) * gamma * x ** (2*b+2)  # Partial derivative of unsaturated hydraulic conductivity of soil in m/d
    unit = 24 * 3600  # 1/s -> 1/d
    grav = 9.81  # gravitational acceleration in N/Kg
    M_w = 0.018  # molar mass of water in Kg/mol
    dgSR = dgx_dx / unit / grav / lSR  # kg/N/s
    dgSR *= 1e6 / M_w  # mol/MPa/m2/s
    dgSR *= unit / lai  # mol/MPa/m2/d
    return dgSR


def dphi_dx(psi_l, psi_crit, psi_x, psi_r, dpsicrit_dx, psi_63, w_exp, Kmax, psi_sat, b):
    x = (psi_x / psi_sat) ** - (1 / b)
    part1 = grl_val(psi_r, psi_63, w_exp, Kmax) / grl_val(psi_l, psi_63, w_exp, Kmax)
    part2 = dpsix_dx(x, psi_sat, b)
    part3 = 1 + (2 * b + 3) * (psi_x - psi_r) / x
    part4 = psi_l / psi_crit ** 2 * dpsicrit_dx
    return - part1 * part2 * part3 / psi_crit + part4


def dB_dgs(cp, k2, psi_l, psi_crit, psi_x, psi_r, VPD, psi_63, w_exp, Kmax, psi_sat,
             gamma, b, d_r, z_r, RAI, lai):
    phi = 1 - psi_l / psi_crit
    part1 = - (cp + k2) / phi ** 2
    part2 = dphi_dgs(psi_l, psi_crit, psi_x, psi_r, VPD, psi_63, w_exp, Kmax, psi_sat,
             gamma, b, d_r, z_r, RAI, lai)
    return part1 * part2


def dB_dx(cp, k2, psi_l, psi_crit, psi_x, psi_r, dpsicrit_dx, psi_63, w_exp, Kmax, psi_sat, b):
    phi = 1 - psi_l / psi_crit
    part1 = - (cp + k2) / phi ** 2
    part2 = dphi_dx(psi_l, psi_crit, psi_x, psi_r, dpsicrit_dx, psi_63, w_exp, Kmax, psi_sat, b)
    return part1 * part2


def dAdgs(t, gs, ca, k1_interp, k2_interp, cp_interp, phi=1):
    """

    :param t: time in days
    :param gs: stomatal conductance in mol m-2 d-1
    :param ca: atmospheric carbon concentration
    :param k1_interp: parameter k1 as a function of time in days mol m-2 d-1
    :param k2_interp: parameter k2 in mol mol-1
    :param cp_interp: carbon compensation point in mol mol-1
    :param phi: degree of divorce between ci and cc
    :return: dA_dgs
    """

    k1 = k1_interp(t)
    k2 = k2_interp(t)
    cp = cp_interp(t)
    B = (cp + k2) / phi

    # part11 = np.zeros(k1.shape)
    # part12 = np.zeros(k1.shape)
    #
    # Nok = np.equal(k1, 0)
    # ok = ~Nok
    # part11[ok] = 2 * (B[ok] + ca - cp[ok]) ** 2 * gs[ok] + 2 * (B[ok] - ca + cp[ok]) * k1[ok]
    # part12[ok] = 2 * np.sqrt((B[ok] + ca - cp[ok]) ** 2 * gs[ok] ** 2 +
    #                          2 * (B[ok] - ca + cp[ok]) * gs[ok] * k1[ok] + k1[ok] ** 2)
    #
    # part11[Nok] = B[Nok] + ca - cp[Nok]
    # part12[Nok] = 1

    part11 = (B + ca - cp) ** 2 * gs + (B - ca + cp) * k1
    part12 = np.sqrt((B + ca - cp) ** 2 * gs ** 2 +
                             2 * (B - ca + cp) * gs * k1 + k1 ** 2)
    part2 = B + ca - cp

    return (1/2) * (part2 * part12 - part11)


def dAdB(t, gs, ca, k1_interp, k2_interp, cp_interp, phi=1):
    """

    :param t: time in days
    :param gs: stomatal conductance in mol m-2 d-1
    :param ca: atmospheric carbon concentration
    :param k1_interp: parameter k1 as a function of time in days mol m-2 d-1
    :param k2_interp: parameter k2 in mol mol-1
    :param cp_interp: carbon compensation point in mol mol-1
    :param phi: degree of divorce between ci and cc
    :return: dA_dgs
    """

    k1 = k1_interp(t)
    k2 = k2_interp(t)
    cp = cp_interp(t)
    B = (cp + k2) / phi

    # part11 = np.zeros(k1.shape)
    # part12 = np.zeros(k1.shape)
    #
    # Nok = np.equal(k1, 0)
    # ok = ~Nok
    #
    # part11[ok] = 2 * (B[ok] + ca - cp[ok]) * gs[ok] ** 2 + 2 * gs[ok] * k1[ok]
    # part12[ok] = 2 * np.sqrt((B[ok] + ca - cp[ok]) ** 2 * gs[ok] ** 2 +
    #                          2 * (B[ok] - ca + cp[ok]) * gs[ok] * k1[ok] + k1[ok] ** 2)
    #
    # part11[Nok] = 0
    # part12[Nok] = 1

    part11 = (B + ca - cp) * gs ** 2 + gs * k1
    part12 = np.sqrt((B + ca - cp) ** 2 * gs ** 2 +
                             2 * (B - ca + cp) * gs * k1 + k1 ** 2)

    part2 = gs

    return (1/2) * (part2 * part12 - part11)


def gs_val_meso(gs, t, ca, k1_interp, k2_interp, cp_interp, psi_lcrit_interp, psi_x,
                 VPDinterp, psi_63, w_exp, Kmax, psi_sat, gamma, b, d_r, z_r, RAI, lai, lam):

    psi_crit = psi_lcrit_interp(psi_x)
    VPD = VPDinterp(t)
    k1 = k1_interp(t)
    k2 = k2_interp(t)
    cp = cp_interp(t)
    x = (psi_x / psi_sat) ** - (1 / b)

    psi_r = 1.6 * gs * VPD / gSR_val(x, gamma, b, d_r, z_r, RAI, lai) + psi_x
    # res = root(lambda psi_l: 1.6 * gs * VPD - Kmax * (psi_63 / w_exp) *
    #                    (gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp) - gammaincc(1 / w_exp, (psi_l / psi_63) ** w_exp)),
    #      psi_r + 0.1, method='hybr')
    # psi_l = res.get('x')
    psi_l_temp = gammainccinv(1 / w_exp,
                         - 1.6 * gs * VPD * w_exp / (gammafunc(1 / w_exp) * Kmax * psi_63) +
                         gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp))

    psi_l = psi_63 * psi_l_temp ** (1 / w_exp)

    phi = 1 - psi_l / psi_crit

    part1 = dAdgs(t, gs, ca, k1_interp, k2_interp, cp_interp, phi)
    part2 = dAdB(t, gs, ca, k1_interp, k2_interp, cp_interp, phi) *\
            dB_dgs(cp, k2, psi_l, psi_crit, psi_x, psi_r, VPD, psi_63, w_exp, Kmax, psi_sat,
             gamma, b, d_r, z_r, RAI, lai)

    B = (cp + k2) / phi

    return part1 + part2 - 1.6 * lam * VPD * np.sqrt((B + ca - cp) ** 2 * gs ** 2 +
                             2 * (B - ca + cp) * gs * k1 + k1 ** 2)


def gs_val_profit(gs, t, ca, k1_interp, k2_interp, cp_interp,
                  trans_max_interp, k_max_interp, k_crit_interp, psi_x,
                 VPDinterp, psi_63, w_exp, Kmax, psi_sat, gamma, b, d_r, z_r, RAI, lai):

    trans_max = trans_max_interp(psi_x)
    k_max = k_max_interp(psi_x)
    k_crit = k_crit_interp(psi_x)
    VPD = VPDinterp(t)
    k1 = k1_interp(t)
    k2 = k2_interp(t)
    cp = cp_interp(t)
    a = 1.6
    x = (psi_x / psi_sat) ** - (1 / b)

    psi_r = a * gs * VPD / gSR_val(x, gamma, b, d_r, z_r, RAI, lai) + psi_x

    psi_l_temp = gammainccinv(1 / w_exp,
                         - 1.6 * gs * VPD * w_exp / (gammafunc(1 / w_exp) * Kmax * psi_63) +
                         gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp))

    psi_l = psi_63 * psi_l_temp ** (1 / w_exp)
    grl_psil = grl_val(psi_l, psi_63, w_exp, Kmax)
    grl_psir = grl_val(psi_r, psi_63, w_exp, Kmax)
    gSR = gSR_val(x, gamma, b, d_r, z_r, RAI, lai)
    dEdpsil = grl_psil * gSR / (grl_psir + gSR)
    gs_max = trans_max / a / VPD
    Amax = A_here(gs_max, ca, k1, k2, cp)

    d2Edpsil2 = dEdpsil * (- (w_exp / psi_63) * (psi_l / psi_63) ** (w_exp - 1) +
                           grl_psir * (w_exp / psi_63) * (psi_r / psi_63) ** (w_exp - 1) * grl_psil / (grl_psir + gSR) ** 2)


    part1 = dAdgs(t, gs, ca, k1_interp, k2_interp, cp_interp, 1) * dEdpsil / (a * VPD)
    part2 = - Amax * d2Edpsil2 / (k_max - k_crit) *\
            np.sqrt((k2 + ca) ** 2 * gs ** 2 + 2 * (k2 - ca + 2 * cp) * gs * k1 + k1 ** 2)
    return part2 - part1



# def lam_min_val_meso(lam, trans_max_interp, t, ca, k1_interp, k2_interp, cp_interp, psi_lcrit_interp,
#                      psi_rcrit_interp, psi_x, VPDinterp, psi_63, w_exp, Kmax, psi_sat,
#                      gamma, b, d_r, z_r, RAI, lai):
#
#     psi_crit = psi_lcrit_interp(psi_x)
#     VPD = VPDinterp(t)
#     k1 = k1_interp(t)
#     k2 = k2_interp(t)
#     cp = cp_interp(t)
#
#     psi_r = psi_rcrit_interp(psi_x)
#
#     psi_l = psi_lcrit_interp(psi_x)
#
#     phi = 1 - psi_l / psi_crit
#
#     gs = trans_max_interp(psi_x) / (1.6 * VPD)
#
#     part1 = dAdgs(t, gs, ca, k1_interp, k2_interp, cp_interp, phi)
#     part2 = dAdB(t, gs, ca, k1_interp, k2_interp, cp_interp, phi) *\
#             dB_dgs(cp, k2, psi_l, psi_crit, psi_x, psi_r, VPD, psi_63, w_exp, Kmax, psi_sat,
#              gamma, b, d_r, z_r, RAI, lai)
#
#     B = (cp + k2) / phi
#
#     return part1 + part2 - 1.6 * lam * VPD * np.sqrt((B + ca - cp) ** 2 * gs ** 2 +
#                              2 * (B - ca + cp) * gs * k1 + k1 ** 2)



def psil_crit_val(psi_l, psi_x, psi_sat, gamma, b, d_r, z_r, RAI, lai, Kmax, psi_63, w_exp, frac=0.05):
    import warnings
    x = (psi_x / psi_sat) ** (-1 / b)
    soil_root = gSR_val(x, gamma, b, d_r, z_r, RAI, lai)
#     with warnings.catch_warnings():
#         warnings.filterwarnings('error', category=RuntimeWarning)
#         try:
#             res_psir = root(lambda psi_r: Kmax * psi_63 * gammafunc(1 / w_exp) / w_exp *
#                                   (gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp) -
#                                    gammaincc(1 / w_exp, (psi_l / psi_63) ** w_exp)) -
#                                   soil_root * (psi_r - psi_x),
#                     (psi_l + psi_x) / 2,
#                method='hybr')
#             psi_r = res_psir.get('x')
#         except RuntimeWarning:
#             import pdb; pdb.set_trace()
#             import sys
#             sys.exit()
    res_psir = root(lambda psi_r: Kmax * psi_63 * gammafunc(1 / w_exp) / w_exp *
                                  (gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp) -
                                   gammaincc(1 / w_exp, (psi_l / psi_63) ** w_exp)) -
                                  soil_root * (psi_r - psi_x),
                    (psi_l + psi_x) / 2,
               method='hybr')
    psi_r = res_psir.get('x')    
    print('psil_crit_val status:' + res_psir.message)
    X_part1 = grl_val(psi_r, psi_63, w_exp, Kmax) + soil_root
    X_part2 = grl_val(psi_x, psi_63, w_exp, Kmax) + soil_root

    X = frac * grl_val(psi_x, psi_63, w_exp, Kmax) / Kmax * X_part1 / X_part2
        

#     import pdb; pdb.set_trace()
    return psi_l - psi_63 * (- np.log(X)) ** (1 / w_exp)
