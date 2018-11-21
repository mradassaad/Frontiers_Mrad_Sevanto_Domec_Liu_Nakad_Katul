import numpy as np
import pandas as pd
from Useful_Funcs import*
from scipy.interpolate import interp1d

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

#%%----------------------------PLANT CONSTANTS-------------------------
n = 0.5  # m3/m3
z_r = 0.3  # m
lai = 1.5
m_w = 0.018  # kg/mol
rho_w = 1000  # kg/m3
nu = lai * m_w / rho_w  # m3/mol

unit0 = 24 * 3600   # 1/s -> 1/d
unit1 = 10 ** 3 * nu / (n * z_r)  # mol/m2 -> mmol/mol
unit2 = 18 * 1e-6  # mol H2O/m2/s ->  m/s
unit3 = 1e6  # Pa -> MPa
unit4 = 273.15  # Degree C -> K
atmP = 0.1013  # atmospheric pressure, MPa

v_opt = 174.33  # umol/m2/s
Hav = 61.21  # kJ/mol
Hdv = 200  # kJ/mol
Topt_v = 37.74 + 273.15  # K

j_opt = 155.76  # umol/m2/s
Haj = 43.79  # kJ/mol
Hdj = 200  # kJ/mol
Topt_j = 32.19 + 273.15  # K


gamma = 0.000  # m/d
c = 1
# vpd = 0.015  # mol/mol
# k = 0.05 * unit0  # mol/m2/day

ca = 350 * 1e-6  # mol/mol
a = 1.6


beta = gamma / (n * z_r)  # 1/d
alpha = nu * a / (n * z_r)  # m2/mol


# ------------------ Soil Properties -----------------

psi_sat = 21.8e-4  # Soil water potential at saturation, MPa
b = 4.9  # other parameter

# ------------------ Plant Stem Properties -------------

psi_63 = 4  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 2  # Weibull exponent
Kmax = 2e-3 * unit0  # Maximum plant stem water conductivity, mol/m2/d/MPa

#%% --------------------- CARBON ASSIMILATION -----------------------
# gc = 0.1 # mol CO2 /m2/s
TEMPfull = np.array(drydown['TEMP'])  # K
RNfull = np.array(drydown['RNET'])  # shortwave radiation on leaves, W/m2
PARfull = RNtoPAR(RNfull)  # umol/m2/s
VPDfull = np.array(drydown['VPD'])  # mol/mol

AvgNbDay = 20

PARavg = PARfull[0:48*AvgNbDay]
PARavg = PARavg.reshape((20, 48))
PARavg = np.average(PARavg, axis=0)

TEMPavg = TEMPfull[0:48*AvgNbDay]
TEMPavg = TEMPavg.reshape((20, 48))
TEMPavg = np.average(TEMPavg, axis=0)

VPDavg = VPDfull[0:48*AvgNbDay]
VPDavg = VPDavg.reshape((20, 48))
VPDavg = np.average(VPDavg, axis=0)

days = 10
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
J = J_val(PAR, Jmax)  # umol/m2/s
J *= 1e-6 * unit0  # mol/m2/d

k1 = J / 4  # mol/m2/d
a2 = Kc * (1 + Oi / Ko)  # mol/mol
k2 = (J / 4) * a2 / Vmax  # mol/mol

VPDinterp = interp1d(t, VPD, kind='cubic')
cp_interp = interp1d(t, cp, kind='cubic')
k1_interp = interp1d(t, k1, kind='cubic')
k2_interp = interp1d(t, k2, kind='cubic')
