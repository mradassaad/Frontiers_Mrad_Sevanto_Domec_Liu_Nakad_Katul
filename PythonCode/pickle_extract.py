import pickle
import numpy as np

env_in = open("../no_WUS/environment", "rb")
environment = pickle.load(env_in)

plant_in = open("../no_WUS/plant_vulnerable", "rb")
plant = pickle.load(plant_in)

soil_in = open("../no_WUS/soil", "rb")
soil = pickle.load(soil_in)

VPDavg = environment['VPDavg']
TEMPavg = environment['TEMPavg']
PARavg = environment['PARavg']
VPDinterp = environment['VPDinterp']
cp_interp = environment['cp_interp']
k1_interp = environment['k1_interp']
k2_interp = environment['k2_interp']
AvgNbDay = environment['AvgNbDay']
days = environment['days']
ca = 350 * 1e-6  # mol/mol
a = 1.6

gamma = soil['gamma']
c = soil['c']
n = soil['n']
z_r = soil['z_r']
d_r = soil['d_r']
RAI = soil['RAI']
beta = soil['beta']
psi_sat = soil['psi_sat']
b = soil['b']
t = np.linspace(0, days, 48 * days)

lai = plant['lai']
nu = plant['nu']
v_opt = plant['v_opt']
Hav = plant['Hav']
Hdv = plant['Hdv']
Topt_v = plant['Topt_v']
j_opt = plant['j_opt']
Haj = plant['Haj']
Hdj = plant['Hdj']
Topt_j = plant['Topt_j']
alpha = plant['alpha']
psi_63 = plant['psi_63']
w_exp = plant['w_exp']
Kmax = plant['Kmax']
reversible = plant['reversible']
trans_max_interp = plant['trans_max_interp']
psi_r_interp = plant['psi_r_interp']
psi_l_interp = plant['psi_l_interp']
k_crit_interp = plant['k_crit_interp']
k_max_interp = plant['k_max_interp']


unit0 = 24 * 3600   # 1/s -> 1/d
unit1 = 10 ** 3 * nu / (n * z_r)  # mol/m2 -> mmol/mol
unit2 = 18 * 1e-6  # mol H2O/m2/s ->  m/s
unit3 = 1e6  # 1/Pa -> 1/MPa
unit4 = 273.15  # Degree C -> K
unit5 = 3.6 * 24 * 9.81  # kg.s.m-3 of water -> m/d
unit6 = 1e-3  # J/Kg of water to MPa
atmP = 0.1013  # atmospheric pressure, MPa
