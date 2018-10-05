import numpy as np
import pandas as pd
from datetime import datetime
import glob
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',size=14)

# user defined parameter groups
class SoilRoot:
    def __init__(self,ksat,psat,b,n,Zr,RAI):
        self.ksat = ksat # saturated conductivity, m/s/MPa
        self.psat = psat # saturated soil water potential, MPa
        self.b = b # nonlinearity in soil water retention curve
        self.n = n # soil porosity
        self.Zr = Zr # rooting depth, m
        self.RAI = RAI # root area index
        self.sfc = (psat/(-0.03))**(1/b)  
        self.sw = (psat/(-3))**(1/b) 

class Xylem:
    def __init__(self,gpmax,p50,aa):
        self.gpmax = gpmax # maximum xylem conductivity, m/s/MPa
        self.p50 = p50 # leaf water pontential at 50% loss of conductnace, MPa
        self.aa = aa # nonlinearity of plant vulnerability curve
     
class Environment:
    def __init__(self,SoilM,RNet,Time,VPD,LAI,):
        self.SoilM = SoilM #soil moisture every 30 minutes
        self.SoilMIni = SoilM(0) #Soil moisture at start of drydown
        self.SoilMEnd = SoilM(-1) #Soil moisture at end of drydown
        self.RNet = RNet #Net radiation every 30 minutes, J/m2/s
        self.Time = Time #Time
        
# constants
a0 = 1.6 # ratio between water and carbon conductances
ca = 400 # atmospheric CO2 concentration, umol/mol
rhow = 1000 # water density, kg/m3
g = 9.81 # gravitational acceleration, m/s2
R = 8.31*1e-3 # Gas constant, kJ/mol/K
UNIT_1 = 18*1e-6 # mol H2O/m2/s ->  m/s
UNIT_2 = 1e6 # Pa -> MPa
UNIT_3 = 273.15 # Degree C -> K


# read FluxNet forcings and MODIS LAI
fill_NA = -9999
nobsinaday = 48 # number of observations in a day
# Info on FluxNet data: http://fluxnet.fluxdata.org/data/aboutdata/data-variables/
def ReadInput(datapath,sitename,latitude): # optimal or full
    fname = glob.glob(datapath+'FLuxNet/FLX_'+sitename+'*.csv')[0]
    lai_fname = datapath+'MODIS_LAI/LAI_'+sitename+'.csv'
    
    df = pd.read_csv(fname)
    varnames = list(df)
    # Select necessary variables, compute average soil moisture, and select valid period
    # Units: TA_F, deg C; SW_IN_F, W/m2'; VPD_F, hPa; P_F, mm
    met = df[['TIMESTAMP_START','SW_IN_F_MDS_QC','TA_F','SW_IN_F','VPD_F','P_F']]
    met.rename(columns={'TA_F':'TEMP','SW_IN_F':'RNET','VPD_F':'VPD','P_F':'P'},inplace=True)
    
    # average soil moisture if multiple measurements
    swcname = [string for string in varnames if string.find('SWC_F_MDS_')>-1 and string.find('QC')<0]
    swcqcname = [string for string in varnames if string.find('SWC_F_MDS_')>-1 and string.find('QC')>-1]
    
    met['SOILM'] = df[swcname].replace(fill_NA,np.nan).mean(axis=1)/100 # volumetric, % to decimal
    met['SWC_F_MDS_AVG_QC'] = df[swcqcname].replace(fill_NA,np.nan).mean(axis=1)
    
    # drop data from extrapolation
    met = met.replace(fill_NA,np.nan)
    met = met[~(met['SW_IN_F_MDS_QC']+met['SWC_F_MDS_AVG_QC']).isna()].reset_index()
    
    # Transform date format and variabel units
    met['TIMESTAMP_START'] = met['TIMESTAMP_START'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d%H%M'))
#    DOY = np.array([itm.timetuple().tm_yday+itm.timetuple().tm_hour/24+itm.timetuple().tm_min/1440 for itm in met['TIMESTAMP_START']])
    
    met['TEMP'] = met['TEMP']+UNIT_3 # deg C -> K
    met['VPD'] = met['VPD']/1013.25 # hPa -> kPa/kPa
    
    # start and end at 00:00 in a day
    t0 = met['TIMESTAMP_START'][0]
    t1 = met['TIMESTAMP_START'][len(met)-1]
    dt_hour = 24/nobsinaday
    dt_min = dt_hour*60
    stid = nobsinaday-int(t0.timetuple().tm_min/dt_min+t0.timetuple().tm_hour/dt_hour)
    etid = len(met)-1-int(t1.timetuple().tm_min/dt_min+t1.timetuple().tm_hour/dt_hour)
    met = met[stid:etid]
    
    # report if dataset is not complete
    if sum(met.isna().sum())>0: print(1-met.isna().sum()/len(met)) 
    
    # read LAI data
    lai = pd.read_csv(lai_fname).replace(np.nan,999)
    lai.rename(columns={'system:time_start': 'TIMESTAMP_START','Lai':'LAI'}, inplace=True)
    lai['TIMESTAMP_START'] = lai['TIMESTAMP_START'].apply(lambda x: datetime.strptime(x+'-00/00','%b %d, %Y-%H/%M'))
    lai['qc0'] = lai['FparLai_QC'].apply(lambda x: int(format(int(x), 'b').zfill(7)[0],2)) # 0: Good quality (main algorithm with or without saturation)
    lai['qc2'] = lai['FparLai_QC'].apply(lambda x: int(format(int(x), 'b').zfill(7)[2],2)) # 0: Detectors apparently fine for up to 50% of channels 1, 2
    lai['qc3'] = lai['FparLai_QC'].apply(lambda x: int(format(int(x), 'b').zfill(7)[3:5],2)) # 0: Significant clouds NOT present
    lai['qc5'] = lai['FparLai_QC'].apply(lambda x: int(format(int(x), 'b').zfill(7)[5:7],2)) # 0: Main (RT) method used with no saturation, best result possible
    
    # Fill and smooth LAI
    lai['LAI'].loc[(lai['qc0']!=0) | (lai['qc2']!=0) | (lai['qc3']!=0) | (lai['qc5']!=0)] = np.nan
    lai['LAI'] = lai['LAI'].interpolate(method='linear')
    lai['LAI'] = savitzky_golay(np.array(lai['LAI']),45,3)/10 # a window size of 45*4/30=6 months
    lai = lai.loc[~lai['LAI'].isna()].reset_index()
    
    # interpolate LAI to the same resolution as meterological forcings
    tt = lai['TIMESTAMP_START']-met['TIMESTAMP_START'].min()
    coarse = np.array(tt.apply(lambda x: x.days))
    fine = np.arange(0,len(met)/nobsinaday,1/nobsinaday) 
    if max(fine)>min(coarse):
        lai_itp = np.interp(fine,coarse,np.array(lai['LAI']))
        lai_qc = np.zeros(lai_itp.shape)
        lai_qc[fine<min(coarse)] = 1 # quality control, flag out extrapolation data
        if min(fine)<min(coarse): # use the LAI in the most recent years if no observation
            st = np.where(fine==min(coarse))[0][0]
            lai_itp[fine<min(coarse)] = np.nan; lai_itp[fine>max(coarse)]=np.nan
            patch = lai_itp[st:(st+int(nobsinaday*362.25))]
            patch = np.tile(patch,[int(np.ceil(sum(fine<min(coarse))/len(patch))),])
            lai_itp[:st] = patch[-sum(fine<min(coarse)):]
    met['LAI'] = lai_itp
    met['LAI_QC'] = lai_qc
    
    # Optional: only use the data when LAI observation is available
    met = met[met['TIMESTAMP_START']>lai['TIMESTAMP_START'][0]]
    return met.reset_index()

def Interstorm(df,drydownid):
    dailyP = dailyAvg(np.array(df['P']),nobsinaday).ravel()
    rainyday = np.where(dailyP>0)[0]
    drydownlength = np.concatenate([np.diff(rainyday),[0]])
    id1 = rainyday[drydownlength>30]+1 # start day of each dry down period longer than 30 days
    id2 = id1+drydownlength[drydownlength>30]-1 # end day of each dry down period
    st = list(df['TIMESTAMP_START'][id1*nobsinaday-1])
    et = list(df['TIMESTAMP_START'][id2*nobsinaday-1])
#    print([st,et])
    print('Selected period: '+str(st[drydownid])+' to '+str(et[drydownid]))
    return df[(df['TIMESTAMP_START']>=st[drydownid]) & (df['TIMESTAMP_START']<et[drydownid])]


# a function to smooth noisy data using polynomial interpolation
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial 
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid') 

def dailyAvg(data,windowsize):
    data = np.array(data)
    data = data[0:windowsize*int(len(data)/windowsize)]
    return np.nanmean(np.reshape(data,[int(len(data)/windowsize),windowsize]),axis=1)

# Compute An as a function of gc under given climate
def f_An(gc,T,RN): # unifts: mol CO2/m2/s, K, W/m2
    
    # photosynthetically active radiation
    hc = 2e-25 # Planck constant times light speed, J*s times m/s
    wavelen = 500e-9 # wavelength of light, m
    EE = hc/wavelen # energy of photon, J
    NA = 6.02e23 # Avogadro's constant, /mol
    PAR = RN/(EE*NA)*1e6 # absorbed photon irradiance, umol photons /m2/s, PAR
    
    # temperature correction
    koptj = 155.76 #  umol/m2/s
    Haj = 43.79 # kJ/mol
    Hdj = 200; # kJ/mol
    Toptj = 32.19+UNIT_3 # K
    koptv = 174.33 # umol/m2/s
    Hav = 61.21 # kJ/mol
    Hdv = 200 # kJ/mol
    Toptv = 37.74+UNIT_3 # K
    Coa = 210 # mmol/mol
    kai1 = 0.9
    kai2 = 0.3
    Vcmax = koptv*Hdv*np.exp(Hav*(T-Toptv)/T/R/Toptv)/(Hdv-Hav*(1-np.exp(Hav*(T-Toptv)/T/R/Toptv))) # umol/m2/s
    Jmax = koptj*Hdj*np.exp(Haj*(T-Toptj)/T/R/Toptj)/(Hdj-Haj*(1-np.exp(Haj*(T-Toptj)/T/R/Toptj)))
    TC = T-UNIT_3 # C
    Kc = 300*np.exp(0.074*(TC-25)) # umol/mol
    Ko = 300*np.exp(0.015*(TC-25)) # mmol/mol
    cp = 36.9+1.18*(TC-25)+0.036*(TC-25)**2
    J = (kai2*PAR+Jmax-np.sqrt((kai2*PAR+Jmax)**2-4*kai1*kai2*PAR*Jmax))/2/kai1 # umol electrons /m2/s
    Rd = 0.015*Vcmax # daytime mitochondrial respiration rate
    
    # solve carbon assimilation based on Fickian diffusion and the Farquhar model
    a1 = J/4;a2 = 2*cp # RuBP limited photosynthesis (light limitation)
    B = (a1-Rd)/gc+a2-ca
    C = -(a1*cp+a2*Rd)/gc-ca*a2
    ci = (-B+np.sqrt(B**2-4*C))/2
    An1 = gc*(ca-ci)
#    An11 = a1*(ci-cp)/(a2+ci)-Rd # check solution
#    plt.plot(An1-An11)
    a1 = Vcmax;a2 = Kc*(1+Coa/Ko) # Rubisco limited photosynthesis
    B = (a1-Rd)/gc+a2-ca
    C = -(a1*cp+a2*Rd)/gc-ca*a2
    ci = (-B+np.sqrt(B**2-4*C))/2
    An2 = gc*(ca-ci)
    An = np.min(np.column_stack([An1,An2]),axis=1)
    An[An<0] = 0
    return An # unit: umol CO2 /m2/s


def VulnerabilityCurve(Xparas,psil):
    return Xparas.gpmax/(1+(psil/Xparas.p50)**Xparas.aa) # m/s

def f_soilroot(s,SRparas):
    K = SRparas.ksat*np.power(s,SRparas.b*2+3) # soil hydraulic conductivity m/s
    psis =  SRparas.psat*np.power(s,-SRparas.b) # soil water potential MPa
    gsr = K*np.sqrt(SRparas.RAI)/(rhow*g*SRparas.Zr*np.pi)*UNIT_2 # m/s
    return psis,gsr

def Opt(Environment,SoilRoot,Xylem):
    
    return
    
#%% -------------------------- READ DATA ----------------------------
# read directly from fluxnet dataset 
datapath = '../Data/'
sitename = 'US-Blo'
latitude = 38.8953 # to be modified if changing site
#df = ReadInput(datapath,sitename,latitude)
#df.to_csv(datapath+'FLX_'+sitename+'.csv')

# read cleaned data 
df = pd.read_csv(datapath+'FLX_'+sitename+'.csv')
drydownid = 2
drydown = Interstorm(df,drydownid) # data during the 2nd dry down period

#%% --------------------- CARBON ASSIMILATION -----------------------
gc = 0.1 # mol CO2 /m2/s
TEMP = np.array(drydown['TEMP'])
RN =  np.array(drydown['RNET']) # shortwave radiation on leaves
An = f_An(gc,TEMP,RN)
plt.figure()
plt.plot(An,'-k');plt.xlim([0,48*5])
plt.xlabel('Time step (half-hour)')
plt.ylabel(r'An ($\mu$mol CO$_2$ /m$^2$/s)')

#%% --------------------- TRANSPIRATION -----------------------------
VPD = np.array(drydown['VPD']) # kPa/kPa
LAI = np.array(drydown['LAI'])
Tr = a0*gc*VPD*LAI*UNIT_1 # m/s
plt.figure()
plt.plot(Tr,'-k');plt.xlim([0,48*5])
plt.xlabel('Time step (half-hour)')
plt.ylabel('Transpiration (m/s)')


#%% ------------------ LEAF WATER POTENTIAL ------------------------
s = 0.5 # relative soil moisture, \in (0,1)
SRparas = SoilRoot(3.5e-4,-0.00696,3.5,0.4,1,10) 
Xparas = Xylem(3e-7,-3,2)
psis,gsr = f_soilroot(s,SRparas)
psir = psis-Tr/gsr # root water potential, assuming steady state, continuity
psil0 = -0.5
psil = np.array([opt.fsolve(lambda x: VulnerabilityCurve(Xparas,x)*
                            (psir[np.where(Tr==tt)]-x)-tt,
                            psil0) for tt in Tr])
plt.figure()
plt.plot(psil,'-k');plt.xlim([0,48*5])
plt.xlabel('Time step (half-hour)')
plt.ylabel('Leaf water potential (MPa)')

