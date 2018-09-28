import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import glob
import pymc as pm


class SoilRoot:
    def __init__(self,ksat,psat,b1,b2,n1,n2,Zr1,Zr2,RAI1,RAI2,z,ch):
        self.ksat1 = ksat
        self.ksat2 = ksat
        self.psat1 = psat
        self.psat2 = psat
        self.b1 = b1
        self.b2 = b2
        self.n1 = n1
        self.n2 = n2
        self.Zr1 = Zr1
        self.Zr2 = Zr2
        self.RAI1 = RAI1
        self.RAI2 = RAI2
        self.sfc1 = (psat/(-0.03))**(1/b1)
        self.sw1 = (psat/(-3))**(1/b1)
        self.sfc2 = (psat/(-0.03))**(1/b2)
        self.sw2 = (psat/(-3))**(1/b2)
        self.z = z # measurement height
        self.ch = ch # canopy height

class XylemLeaf:
    def __init__(self,gpmax,p50,aa,lww,b0,Cx,Cl):
        self.gpmax = gpmax
        self.p50 = p50
        self.aa = aa
        self.lww = lww
        self.b0 = b0
        self.Cx = Cx
        self.Cl = Cl

class InitialState:
    def __init__(self,psir,psix,psil,pisl_avg,s1,s2):
        self.psir = psir
        self.psix = psix
        self.psil = psil
        self.psil_avg = pisl_avg
        self.s1 = s1
        self.s2 = s2

class Climate:
    def __init__(self,temp,vpd,rnet,lai):
        self.temp = temp
        self.vpd = vpd
        self.rnet = rnet
        self.lai = lai
        
rhow = 1e3; g = 9.81
a0 = 1.6; 
UNIT_1 = a0*18*1e-6 # mol CO2 /m2/s -> m/s, H2O
UNIT_2 = 1e6 # Pa -> MPa
UNIT_3 = 273.15 # Degree C -> K
CCp = 1005; rhoa = 1.225; lelambda = 2450000; p0 = 101325 # Pa
R = 8.31*1e-3 # Gas constant, kJ/mol/K
gamma= CCp*p0/0.622/lelambda # Pa/K
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
ca = 400; ca0 = 400
ga = 0.02
dt = 60*30 # s -> 30 min

karman = 0.4
g = 9.81
rhohat = 44.6 # mol/m3
Cpmol = 1005*28.97*1e-3 # J/kg/K*kg/mol -> J/mol/K
lambdamol = 40660 # J/mol

fill_NA = -9999
nobsinaday = 48 # number of observations in a day
leaf_angle_distr = 1 # spherical distribution for light extinction 


# read FluxNet forcings and MODIS LAI
# Info on FluxNet data: 
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
    DOY = np.array([itm.timetuple().tm_yday+itm.timetuple().tm_hour/24+itm.timetuple().tm_min/1440 for itm in met['TIMESTAMP_START']])
    
    met['Vegk'] = LightExtinction(DOY,latitude,leaf_angle_distr)
    met['TEMP'] = met['TEMP']+273.15 # deg C -> K
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

def LightExtinction(DOY,lat,x):
    B = (DOY-81)*2*np.pi/365
    ET = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
    DA = 23.45*np.sin((284+DOY)*2*np.pi/365)# Deviation angle
    LST = np.mod(DOY*24*60,24*60)
    AST = LST+ET
    h = (AST-12*60)/4 # hour angle
    alpha = np.arcsin((np.sin(np.pi/180*lat)*np.sin(np.pi/180*DA)+np.cos(np.pi/180*lat)*np.cos(np.pi/180.*DA)*np.cos(np.pi/180*h)))*180/np.pi # solar altitude
    zenith_angle = 90-alpha
    Vegk = np.sqrt(x**2+np.tan(zenith_angle/180*np.pi)**2)/(x+1.774*(1+1.182)**(-0.733)) # Campbell and Norman 1998
    return Vegk

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


datapath = '../Data/'
sitename = 'US-Blo'
latitude = 38.8953 # to be modified if changing site

drydownid = 2
df = ReadInput(datapath,sitename,latitude)
drydown = Interstorm(df,drydownid)

#%%


