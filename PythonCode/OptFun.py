import numpy as np
import numpy.ma as ma
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime
import glob




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


def J_val(PAR):
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


def A(gl):
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

    A = np.zeros(gl.size)
    gl_mask = ma.masked_less_equal(gl,0)
    gl_valid = gl[~gl_mask.mask]

    k1 = J[~gl_mask.mask]/4  # mol/m2/d
    a2 = Kc[~gl_mask.mask] * (1 + Oi/Ko[~gl_mask.mask])  # mol/mol
    k2 = (J[~gl_mask.mask] / 4) * a2 / Vmax[~gl_mask.mask]  # mol/mol
    delta = np.sqrt((k2 + ca + k1/gl_valid) ** 2 -
                    4 * k1 * (ca - cp[~gl_mask.mask])  / gl_valid)  # mol/mol

    A[~gl_mask.mask] = 0.5 * (k1 + gl_valid * (ca + k2) - gl_valid * delta)  # mol/m2/d
    # A *= 1e6/unit0


    return A



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

def g_val(lam):

    gpart11 = np.sqrt(alpha * lam * k1_interp(t)**2 * VPDinterp(t) * (ca - cp_interp(t)) * (cp_interp(t) + k2_interp(t)) *
                     ((ca + k2_interp(t)) - 2 * alpha * lam * VPDinterp(t))**2 *
                     ((ca + k2_interp(t)) - alpha * lam * alpha))  # mol/m2/d

    gpart12 = alpha * lam * VPDinterp(t) * ((ca + k2_interp(t)) - alpha * lam * VPDinterp(t))  # mol/mol

    gpart21 = gpart11 / gpart12  # mol/m2/d

    gpart22 = (ca * k1_interp(t) - k1_interp(t) * (2 * cp_interp(t) + k2_interp(t)))  # mol/m2/d

    gpart3 = gpart21 + gpart22  # mol/m2/d

    gpart4 = (ca + k2_interp(t))**2  # mol2/mol2

    g = gpart3 / gpart4  # mol/m2/d

    return g


def dydt(t, y):
    """

    :param t: time in 30 min intervals
    :param y: y[0] is lambda(t) in mol/m2, y[1] is x(t) in mol/mol
    :return:
    """

    gpart11 = np.sqrt(alpha * y[0] * k1_interp(t)**2 * VPDinterp(t) * (ca - cp_interp(t)) * (cp_interp(t) + k2_interp(t)) *
                     ((ca + k2_interp(t)) - 2 * alpha * y[0] * VPDinterp(t))**2 *
                     ((ca + k2_interp(t)) - alpha * y[0] * alpha))  # mol/m2/d

    gpart12 = alpha * y[0] * VPDinterp(t) * ((ca + k2_interp(t)) - alpha * y[0] * VPDinterp(t))  # mol/mol

    gpart21 = gpart11 / gpart12  # mol/m2/d

    gpart22 = (ca * k1_interp(t) - k1_interp(t) * (2 * cp_interp(t) + k2_interp(t)))  # mol/m2/d

    gpart3 = gpart21 + gpart22  # mol/m2/d

    gpart4 = (ca + k2_interp(t))**2  # mol2/mol2

    g = gpart3 / gpart4  # mol/m2/d

    losses = beta * y[1]**c  # 1/d
    evap_trans = alpha * g * VPDinterp(t)  # 1/d
    f = - (losses + evap_trans)  # 1/d

    dlamdt = y[0] * beta * c * y[1]**(c-1)  # mol/m2/d
    dxdt = f  # 1/d

    return np.vstack((dlamdt, dxdt))


def Weibull(Xylem, psi_s, psi_l):

    return Xylem.gpmax * (np.exp(((psi_s - psi_l) / Xylem.psi_63) ** Xylem.c))



# read FluxNet forcings and MODIS LAI
fill_NA = -9999
nobsinaday = 48  # number of observations in a day
# Info on FluxNet data: http://fluxnet.fluxdata.org/data/aboutdata/data-variables/

#%% -------------------------- READ ENVIRONMENTAL DATA ----------------------------
# read directly from fluxnet dataset
datapath = '../Data/'
sitename = 'US-Blo'
latitude = 38.8953  # to be modified if changing site
#df = ReadInput(datapath,sitename,latitude)
#df.to_csv(datapath+'FLX_'+sitename+'.csv')

# read cleaned data
df = pd.read_csv(datapath+'FLX_'+sitename+'.csv')
drydownid = 2
drydown = Interstorm(df, drydownid) # data during the 2nd dry down period


Oi = 210e-3  # mol/mol

#%%----------------------------PLANT CONSTANTS-------------------------
n = 0.5  # m3/m3
z_r = 0.3  # m
lai = 1.5
m_w = 0.018  # kg/mol
rho_w = 1000  # kg/m3
t_day = 1  # day/day
nu = lai * m_w * t_day / rho_w  # m3/mol

unit0 = 24 * 3600 * t_day  # 1/s -> 1/d
unit1 = 10 ** 3 * nu / (n * z_r)  # mol/m2 -> mmol/mol
unit2 = 18 * 1e-6  # mol H2O/m2/s ->  m/s
unit3 = 1e6  # Pa -> MPa
unit4 = 273.15  # Degree C -> K

v_opt = 174.33  # umol/m2/s
Hav = 61.21  # kJ/mol
Hdv = 200  # kJ/mol
Topt_v = 37.74 + 273.15  # K

j_opt = 155.76  # umol/m2/s
Haj = 43.79  # kJ/mol
Hdj = 200  # kJ/mol
Topt_j = 32.19 + 273.15  # K


gamma = 0.01  # m/d
# vpd = 0.015  # mol/mol
# k = 0.05 * unit0  # mol/m2/day

ca = 350 * 1e-6 # mol/mol
a = 1.6

c = 1
beta = gamma / (n * z_r)  # 1/d
alpha = nu * a / (n * z_r)  # m2/mol



#%% --------------------- CARBON ASSIMILATION -----------------------
# gc = 0.1 # mol CO2 /m2/s
TEMPfull = np.array(drydown['TEMP'])  # K
RNfull = np.array(drydown['RNET'])  # shortwave radiation on leaves, W/m2
PARfull = RNtoPAR(RNfull)  # umol/m2/s
VPDfull = np.array(drydown['VPD'])  # mol/mol

days = 20
tlen = 48 * days

t = np.linspace(0, days, tlen)

TEMP = TEMPfull[0:tlen]
PAR = PARfull[0:tlen]
VPD = VPDfull[0:tlen]

Kc, Ko = MMcoeff(TEMP)  # umol/mol and mmol/mol, respectively
Kc *= 1e-6  # mol/mol
Ko *= 1e-3  # mol/mol

cp = cp_val(TEMP)  # umol/mol
cp *= 1e-6  # mol/mol

Vmax = max_val(v_opt, Hav, Hdv, TEMP, Topt_v)  # umol/m2/s
Vmax *= 1e-6 * unit0  # mol/m2/d

Jmax = max_val(j_opt, Haj, Hdj, TEMP, Topt_j)  # umol/m2/s
J = J_val(PAR)  # umol/m2/s
J *= 1e-6 * unit0  # mol/m2/d

k1 = J / 4  # mol/m2/d
a2 = Kc * (1 + Oi / Ko)  # mol/mol
k2 = (J / 4) * a2 / Vmax  # mol/mol

VPDinterp = interp1d(t, VPD, kind='cubic')
cp_interp = interp1d(t, cp, kind='cubic')
k1_interp = interp1d(t, k1, kind='cubic')
k2_interp = interp1d(t, k2, kind='cubic')

# ------------------------OPT Boundary Conditions----------------


def bc(ya, yb):  # boundary imposed on x at t=T
    x0 = 0.8
    return np.array([ya[1] - x0, yb[1]])


def bc_wus(ya,yb):  # Water use strategy
    x0 = 0.8
    wus_coeff = 800e-6*t_day*unit0  # mol/m2
    return np.array([ya[1] - x0, yb[0] - wus_coeff])


lam_guess = 1*np.ones((1, t.size))
x_guess = 0.6*np.ones((1, t.size))

y_guess = np.vstack((lam_guess, x_guess))

res = solve_bvp(dydt, bc, t, y_guess)

lam_plot = res.sol(t)[0]*unit1
soilM_plot = res.sol(t)[1]

A_val = A(g_val(res.sol(t)[0]))

plt.figure()
plt.subplot(221)
plt.plot(t, lam_plot)
#plt.xlabel("days")
plt.ylabel("$\lambda (t), mmol.mol^{-1}$")

plt.subplot(222)
plt.plot(t, soilM_plot)
# plt.xlabel("time, days")
plt.ylabel("$x(t)$")

plt.subplot(223)
plt.plot(t, g_val(res.sol(t)[0]) / unit0)
plt.xlabel("time, days")
plt.ylabel("$g(t), mol.m^{-2}.s^{-1}$")



plt.subplot(224)
plt.plot(t, A(g_val(res.sol(t)[0])) * 1e6 / unit0)
plt.xlabel("time, days")
plt.ylabel("$A, \mu mol.m^{-2}.s^{-1}$")

plt.figure()
plt.subplot(311)
plt.plot(t, VPD)
#plt.xlabel("days")
plt.ylabel("$VPD, mol/mol$")

plt.subplot(312)
plt.plot(t, TEMP)
plt.ylabel("$T, K$")

plt.subplot(313)
plt.plot(t, PAR)
plt.xlabel("time, days")
plt.ylabel("$PAR, umol.m^{-2}.s^{-1}$")

