import numpy as np
from scipy import optimize

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

    :param: k_opt: value of either J_max or V_cmax at Topt in umol/m2/s
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


def RNtoPAR(RN):
    """

    :param RN: radiation in W/m2
    :return: PAR in umol/m2/s
    """
    hc = 2e-25  # Planck constant times light speed, J*s times m/s
    wavelen = 500e-9  # wavelength of light, m
    EE = hc / wavelen  # energy of photon, J
    NA = 6.02e23  # Avogadro's constant, /mol
    PAR = RN / (EE * NA) * 1e6  # absorbed photon irradiance, umol photons /m2/s, PAR
    return PAR


def J_val(PAR, Jmax):
    """

    :param PAR: photosynthetically active photon flux density in umol/m2/s
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


def A(t, gl, ca, k1_interp, k2_interp, cp_interp):
    """

    :param gl: stomatal conductance in umol/m2/s
    :return: value of A at a particular value of g in umol/m2/s
    """
    '''J: Electron transport rate in umol/m2/s
     Vc_max: maximum rate of rubisco activity in umol/m2/s
    Kc: Michaelis-Menten constant for CO2 in umol/mol
     Ko: Michaelis-Menten constant for O2 in mmol/mol
     ca: ambient CO2 mole fraction in the air in umol/mol
    cp: CO2 concentration at which assimilation is zero or compensation point in umol/mol'''

    # A = np.zeros(gl.size)
    # gl_mask = ma.masked_less_equal(gl, 0)
    # gl_valid = gl[~gl_mask.mask]
    #
    # k1 = J[~gl_mask.mask]/4  # mol/m2/d
    # a2 = Kc[~gl_mask.mask] * (1 + Oi/Ko[~gl_mask.mask])  # mol/mol
    # k2 = (J[~gl_mask.mask] / 4) * a2 / Vmax[~gl_mask.mask]  # mol/mol
    # delta = np.sqrt((k2 + ca + k1/gl_valid) ** 2 -
    #                 4 * k1 * (ca - cp[~gl_mask.mask]) / gl_valid)  # mol/mol
    #
    # A[~gl_mask.mask] = 0.5 * (k1 + gl_valid * (ca + k2) - gl_valid * delta)  # mol/m2/d

    delta = np.sqrt(((k2_interp(t) + ca) * gl + k1_interp(t)) ** 2 -
                    4 * k1_interp(t) * (ca - cp_interp(t)) * gl)  # mol/mol

    A = 0.5 * (k1_interp(t) + gl * (ca + k2_interp(t)) - delta)  # mol/m2/d
    # A *= 1e6/unit0


    return A


# def soil_water_pot(x, b, psi_sat):
#
#     return psi_sat * x ** (-b)

# def dAdg(J, Vc_max, Kc, Ko, Oi, ca, cp, gl):
#     """
#
#     :param J: Electron transport rate in umol/m2/s
#     :param Vc_max: maximum rate of rubisco activity in umol/m2/s
#     :param Kc: Michaelis-Menten constant for CO2 in umol/mol
#     :param Ko: Michaelis-Menten constant for O2 in mmol/mol
#     :param ca: ambient CO2 mole fraction in the air in umol/mol
#     :param cp: CO2 concentration at which assimilation is zero or compensation point in umol/mol
#     :param gl: stomatal conductance in umol/m2/s
#     :return: value of A at a particular value of g in umol/m2/s
#     """
#
#     k1 = J / 4  # umol/m2/s
#     a2 = Kc * (1 + Oi / Ko)  # umol/mol
#     k2 = (J / 4) * a2 / Vc_max  # umol/mol
#     ddeltadg = 2 * (k1 / gl**2) * (k2*1e-6 + ca * 1e-6 + k1/gl) + \
#         (4 * k1 / gl**2) * (ca - cp) * 1e-6
#
#     return

def Interstorm(df, drydownid):
    nobsinaday = 48  # number of observations in a day

    dailyP = dailyAvg(np.array(df['P']), nobsinaday).ravel()
    rainyday = np.where(dailyP > 0)[0]
    drydownlength = np.concatenate([np.diff(rainyday), [0]])
    id1 = rainyday[drydownlength > 30]+1  # start day of each dry down period longer than 30 days
    id2 = id1+drydownlength[drydownlength > 30]-1  # end day of each dry down period
    st = list(df['TIMESTAMP_START'][id1*nobsinaday-1])
    et = list(df['TIMESTAMP_START'][id2*nobsinaday-1])
#    print([st,et])
    print('Selected period: '+str(st[drydownid])+' to '+str(et[drydownid]))
    return df[(df['TIMESTAMP_START'] >=
               st[drydownid]) & (df['TIMESTAMP_START'] < et[drydownid])]


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

def g_val(t, lam, ca, alpha, VPDinterp, k1_interp, k2_interp, cp_interp):

    gpart11 = (ca + k2_interp(t) - 2 * alpha * lam * VPDinterp(t)) *\
              np.sqrt(alpha * lam * VPDinterp(t) * (ca - cp_interp(t)) * (cp_interp(t) + k2_interp(t)) *
              (ca + k2_interp(t) - alpha * lam * VPDinterp(t)))  # mol/m2/d

    gpart12 = alpha * lam * VPDinterp(t) * ((ca + k2_interp(t)) - alpha * lam * VPDinterp(t))  # mol/mol

    gpart21 = gpart11 / gpart12  # mol/m2/d

    gpart22 = (ca - (2 * cp_interp(t) + k2_interp(t)))  # mol/m2/d

    gpart3 = gpart21 + gpart22  # mol/m2/d

    gpart4 = (ca + k2_interp(t))**2  # mol2/mol2

    gl = k1_interp(t) * gpart3 / gpart4  # mol/m2/d

    return gl


def psil_val(psi_l, psi_x, psi_63, w_exp, Kmax, gl, lai, VPDinterp, t):
    # psi_l_mask = ma.masked_greater_equal(psi_l, psi_x)
    # f_psi_l = np.zeros(psi_l.size)
    # temp = psi_l - lai * gl * VPDinterp(t) / (Kmax * np.exp(- (0.5 * (psi_x + psi_l) / psi_63) ** w_exp)) - psi_x
    #
    # f_psi_l[psi_l_mask.mask] = temp[psi_l_mask.mask]

    if np.any(
            np.logical_not(np.isfinite(transpiration(psi_l, psi_x, psi_63, w_exp, Kmax) - lai * gl * VPDinterp(t)))):
        raise GuessError('Try increasing the lambda guess or there may be no solutions for the parameter choices.')

    return transpiration(psi_l, psi_x, psi_63, w_exp, Kmax) - lai * gl * VPDinterp(t)
    # return f_psi_l


def transpiration(psi_l, psi_x, psi_63, w_exp, Kmax):
    #  returns transpiration in mol/m2/d
    return (psi_l - psi_x) * (Kmax * np.exp(- (0.5 * (psi_x + psi_l) / psi_63) ** w_exp))


def trans_opt(psi_l, psi_x, psi_63, w_exp, Kmax):

    return - transpiration(psi_l, psi_x, psi_63, w_exp, Kmax)
