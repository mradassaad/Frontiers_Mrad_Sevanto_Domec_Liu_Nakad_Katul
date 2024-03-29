import numpy as np
import pandas as pd
from Useful_Funcs import*
from scipy.interpolate import interp1d
from scipy.optimize import minimize

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
drydown = Interstorm(df, drydownid)  # data during the 2nd dry down period

Oi = 210e-3  # mol/mol
# vpd = 0.015  # mol/mol
# k = 0.05 * unit0  # mol/m2/day

ca = 350 * 1e-6  # mol/mol
a = 1.6
#%%----------------------------PLANT CONSTANTS-------------------------
n = 0.5  # m3/m3
d_r = 1e-3  # fine root diameter in meters
z_r = 1  # m
RAI = 10  # root area index
lai = 1.5
m_w = 0.018  # kg/mol
rho_w = 1000  # kg/m3
nu = lai * m_w / rho_w  # m3/mol

unit0 = 24 * 3600   # 1/s -> 1/d
unit1 = 10 ** 3 * nu / (n * z_r)  # mol/m2 -> mmol/mol
unit2 = 18 * 1e-6  # mol H2O/m2/s ->  m/s
unit3 = 1e6  # 1/Pa -> 1/MPa
unit4 = 273.15  # Degree C -> K
unit5 = 3.6 * 24 * 9.81  # kg.s.m-3 of water -> m/d
unit6 = 1e-3  # J/Kg of water to MPa
atmP = 0.1013  # atmospheric pressure, MPa

# Medlyn 2002 Pinus pinaster but kopt is altered so that k25 matches the values for ponderosa pine in Panek 2004
v_opt = 85.4  # umol/m2/s per LEAF area
Hav = 74.16  # kJ/mol
Hdv = 200  # kJ/mol
Topt_v = 38.34 + 273.15  # K

j_opt = 114.23  # umol/m2/s per LEAF area
Haj = 34.83  # kJ/mol
Hdj = 200  # kJ/mol
Topt_j = 36.87 + 273.15  # K

# ------------------ Soil Properties -----------------

psi_sat = 1.5 * unit6  # Soil water potential at saturation, MPa
b = 3.1  # other parameter

gamma = 0.000  # m/d per unit ground area
c = 1

# --- Using the Campbell(1974) equations, comment out next two lines if don't want

gamma = 0.00072 * unit5   # m/d, for sandy loam page 130 Campbell and Norman
c = 2*b+3

beta = gamma / (n * z_r)  # 1/d
alpha = nu * a / (n * z_r)  # m2/mol

# ------------------ Plant Stem Properties -------------
# psi_63=4.3 MPa and s=5.4 for ponderosa pine according to Hubbard et al. PCE 2001 grown in Utah nurseries
# grown between 20 and 28 degrees C and 40-60% humidity
# Kmax assumed that we have 20 mmol m-1 s-1 MPa-1 for a 10 m tall tree
psi_63 = 4.3  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 2  # Weibull exponent
Kmax = 2e-3 * unit0  # Maximum plant stem water leaf area-averaged conductivity, mol/m2/d/MPa
reversible = 0
# ----------------- Compute transpiration maxima -----------

xvals = np.arange(0.06, 0.8, 0.005)
psi_x_vals = psi_sat * xvals ** -b
soil_root = gSR_val(xvals, gamma, b, d_r, z_r, RAI, lai)

res = root(psil_crit_val, psi_x_vals + 0.000001,
                                args=(psi_x_vals, psi_sat, gamma, b, d_r, z_r, RAI, lai, Kmax, psi_63, w_exp),
                                method='hybr')
psi_l_vals = res.get('x')

res_psir = root(lambda psi_r: Kmax * psi_63 * gammafunc(1 / w_exp) / w_exp *
                                  (gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp) -
                                   gammaincc(1 / w_exp, (psi_l_vals / psi_63) ** w_exp)) -
                                  soil_root * (psi_r - psi_x_vals),
                    (psi_l_vals + psi_x_vals) / 2,
               method='hybr')
psi_r_vals = res_psir.get('x')

trans_vals = soil_root * (psi_r_vals - psi_x_vals)
k_max_vals = grl_val(psi_x_vals, psi_63, w_exp, Kmax) * soil_root /\
             (grl_val(psi_x_vals, psi_63, w_exp, Kmax) + soil_root)
k_crit_vals = 0.05 * k_max_vals

dpsil_dx_vals = np.gradient(psi_l_vals, xvals)
dpsir_dx_vals = np.gradient(psi_r_vals, xvals)

trans_max_interp = interp1d(psi_x_vals, trans_vals, kind='cubic', bounds_error=False, fill_value=(trans_vals[-1],0))  # per unit LEAF area
psi_l_interp = interp1d(psi_x_vals, psi_l_vals, kind='cubic', bounds_error=False, fill_value=(psi_l_vals[-1],psi_l_vals[0]))
psi_r_interp = interp1d(psi_x_vals, psi_r_vals, kind='cubic', bounds_error=False, fill_value=(psi_r_vals[-1],psi_r_vals[0]))
k_crit_interp = interp1d(psi_x_vals, k_crit_vals, kind='cubic', bounds_error=False, fill_value=(k_crit_vals[-1],0))
k_max_interp = interp1d(psi_x_vals, k_max_vals, kind='cubic', bounds_error=False, fill_value=(k_max_vals[-1],0))

dpsil_dx_interp = interp1d(psi_x_vals, dpsil_dx_vals, kind='cubic')
dpsir_dx_interp = interp1d(psi_x_vals, dpsir_dx_vals, kind='cubic')

dgSR_dx = np.gradient(gSR_val(xvals, gamma, b, d_r, z_r, RAI, lai), xvals)
dgSR_dx_interp = interp1d(xvals, dgSR_dx, kind='cubic')

#%% --------------------- CARBON ASSIMILATION -----------------------
# gc = 0.1 # mol CO2 /m2/s
TEMPfull = np.array(drydown['TEMP'])  # K
RNfull = np.array(drydown['RNET'])  # shortwave radiation on leaves, W/m2
PARfull = RNtoPAR(RNfull) / lai # umol/m2/s per unit leaf area
VPDfull = np.array(drydown['VPD'])  # mol/mol

AvgNbDay = 100

PARavg = PARfull[0:48*AvgNbDay]
PARavg = PARavg.reshape((AvgNbDay, 48))
PARavg = np.average(PARavg, axis=0)

TEMPavg = TEMPfull[0:48*AvgNbDay]
TEMPavg = TEMPavg.reshape((AvgNbDay, 48))
TEMPavg = np.average(TEMPavg, axis=0)

VPDavg = VPDfull[0:48*AvgNbDay]
VPDavg = VPDavg.reshape((AvgNbDay, 48))
VPDavg = np.average(VPDavg, axis=0)

days = 110
tlen = 48 * days

t = np.linspace(0, days, tlen)

TEMP = np.tile(TEMPavg, days)
PAR = np.tile(PARavg, days)
VPD = np.tile(VPDavg, days)

Kc, Ko = MMcoeff(TEMP)  # umol/mol and mmol/mol, respectively
Kc *= 1e-6  # mol/mol
Ko *= 1e-3  # mol/mol

cp = cp_val(TEMP)  # umol/mol
cp *= 1e-6  # mol/mol

Vmax = max_val(v_opt, Hav, Hdv, TEMP, Topt_v)  # umol/m2/s
Vmax *= 1e-6 * unit0  # mol/m2/d

Jmax = max_val(j_opt, Haj, Hdj, TEMP, Topt_j)  # umol/m2/s
J = J_val(PAR, Jmax)  # umol/m2/s per unit LEAF area
J *= 1e-6 * unit0  # mol/m2/d

k1 = J / 4  # mol/m2/d
a2 = Kc * (1 + Oi / Ko)  # mol/mol
k2 = (J / 4) * a2 / Vmax  # mol/mol

VPDinterp = interp1d(t, VPD, kind='cubic')
cp_interp = interp1d(t, cp, kind='cubic')
k1_interp = interp1d(t, k1, kind='cubic')
k2_interp = interp1d(t, k2, kind='cubic')

env = {'VPDavg': VPDavg, 'TEMPavg': TEMPavg, 'PARavg': PARavg, 'VPDinterp': VPDinterp,
       'cp_interp': cp_interp, 'k1_interp': k1_interp, 'k2_interp': k2_interp, 'AvgNbDay': AvgNbDay,
       'days': days, 'ca': ca}

soil = {'Soil_type': "Sandy Loam", 'gamma': gamma, 'c': c, 'n': n, 'z_r': z_r, 'd_r': d_r, 'RAI': RAI,
        'beta': beta, 'psi_sat': psi_sat, 'b': b}

plant = {'Plant_type': "Pinus radiata fert.", 'lai': lai, 'nu': nu, 'v_opt': v_opt, 'Hav': Hav,
         'Hdv': Hdv, 'Topt_v': Topt_v, 'j_opt': j_opt, 'Haj': Haj, 'Hdj': Hdj, 'Topt_j': Topt_j,
         'alpha': alpha, 'psi_63': psi_63, 'w_exp': w_exp, 'Kmax': Kmax, 'reversible': reversible,
         'trans_max_interp': trans_max_interp, 'psi_r_interp': psi_r_interp, 'psi_l_interp': psi_l_interp,
         'k_crit_interp': k_crit_interp, 'k_max_interp': k_max_interp, 'dpsil_dx_interp': dpsil_dx_interp,
         'dpsir_dx_interp': dpsir_dx_interp}

import pickle

pickle_out = open("../Long_Drought/environment", "wb")
pickle.dump(env, pickle_out)
pickle_out.close()

pickle_out = open("../Long_Drought/soil", "wb")
pickle.dump(soil, pickle_out)
pickle_out.close()

pickle_out = open("../Long_Drought/plant_ponderosa", "wb")
pickle.dump(plant, pickle_out)
pickle_out.close()

