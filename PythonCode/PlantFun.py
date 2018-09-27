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

def ReadInput(fname,subset): # optimal or full
    sitename = fname[fname.find('\\FLX_')+5:fname.find('_FLUXNET2015')]
    lai_fname = fname[:fname.find('\\FLX_')]+'/LAI/LAI_'+sitename+'.csv'
#    sitename = fname[fname.find('/FLX_')+5:fname.find('_FLUXNET2015')]
#    lai_fname = fname[:fname.find('/FLX_')]+'/LAI/LAI_'+sitename+'.csv'
    fill_NA = -9999
    nobsinaday = 48
    latitude = 38.8953
    leaf_angle_distr = 1 # spherical distribution
    df = pd.read_csv(fname)
    varnames = list(df)
    
    # Select necessary variables, compute average soil moisture, and select valid period
    met = df[['TIMESTAMP_START','LE_F_MDS_QC','SW_IN_F_MDS_QC','TA_F','SW_IN_F','VPD_F','P_F','H_CORR','LE_CORR','WS_F','USTAR','GPP_NT_VUT_REF','NEE_VUT_REF']]
    met.rename(columns={'TA_F':'TEMP','SW_IN_F':'RNET','VPD_F':'VPD','P_F':'P',
                        'SWC_F_MDS_AVG':'SOILM','H_CORR':'H','LE_CORR':'LE','WS_F':'WS',
                        'GPP_NT_VUT_REF':'GPP','NEE_VUT_REF':'NEE'},inplace=True)
    
    swcname = [string for string in varnames if string.find('SWC_F_MDS_')>-1 and string.find('QC')<0]
    swcqcname = [string for string in varnames if string.find('SWC_F_MDS_')>-1 and string.find('QC')>-1]
    
    met['SOILM'] = df[swcname].replace(fill_NA,np.nan).mean(axis=1)/100
    met['SWC_F_MDS_AVG_QC'] = df[swcqcname].replace(fill_NA,np.nan).mean(axis=1)
    
    met = met.replace(fill_NA,np.nan)
    met = met[~(met['LE_F_MDS_QC']+met['SW_IN_F_MDS_QC']+met['SWC_F_MDS_AVG_QC']).isna()].reset_index()
    
    # Transform unit
    met['TIMESTAMP_START'] = met['TIMESTAMP_START'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d%H%M'))
    DOY = np.array([itm.timetuple().tm_yday+itm.timetuple().tm_hour/24+itm.timetuple().tm_min/1440 for itm in met['TIMESTAMP_START']])
    met['Vegk'] = LightExtinction(DOY,latitude,leaf_angle_distr)
    met['TEMP'] = met['TEMP']+273.15
    met['VPD'] = met['VPD']/1013.25
    t0 = met['TIMESTAMP_START'][0]
    t1 = met['TIMESTAMP_START'][len(met)-1]
    stid = 48-int(t0.timetuple().tm_min/30+t0.timetuple().tm_hour*2)
    etid = len(met)-1-int(t1.timetuple().tm_min/30+t1.timetuple().tm_hour*2)
    met = met[stid:etid]
    # Interpolate non-filled variabels
    #met['SWC_F_MDS_AVG'] = met['SWC_F_MDS_AVG'].interpolate(method='linear')/100
    met['USTAR'][met['USTAR'].isna()] = np.nanmean(met['USTAR'])
    if sum(met.isna().sum())>0: print(1-met.isna().sum()/len(met)) # report if dataset is not complete
    
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
    tt = lai['TIMESTAMP_START']-met['TIMESTAMP_START'].min()
    coarse = np.array(tt.apply(lambda x: x.days))
    fine = np.arange(0,len(met)/nobsinaday,1/nobsinaday) 
    if max(fine)>min(coarse):
        lai_itp = np.interp(fine,coarse,np.array(lai['LAI']))
        lai_qc = np.zeros(lai_itp.shape)
        lai_qc[fine<min(coarse)] = 1
        if min(fine)<min(coarse):
            st = np.where(fine==min(coarse))[0][0]
            lai_itp[fine<min(coarse)] = np.nan; lai_itp[fine>max(coarse)]=np.nan
            patch = lai_itp[st:(st+int(nobsinaday*362.25))]
            patch = np.tile(patch,[int(np.ceil(sum(fine<min(coarse))/len(patch))),])
            lai_itp[:st] = patch[-sum(fine<min(coarse)):]
    met['LAI'] = lai_itp
    met['LAI_QC'] = lai_qc
        
    if subset!='full':
        yy = FindTargetYear(met,0.25,0.75,1) #  select the best year for parameter retrieval
        # Return the same dataframe as the previous function DataFilter()
        met = met.loc[(met['TIMESTAMP_START'] >= datetime(yy,10,1,0,0)) & (met['TIMESTAMP_START'] <= datetime(yy+1,9,30,23,30))]
    
    
    met['ET'] = met['LE']/lambdamol*1e3 # mmol/m2/s
    met['ET'] = savitzky_golay(np.array(met['ET']), 12, 3)
    met['ET'].loc[(met['RNET']<=0) | (met['ET']<0)] = 0
    met['GPP'].loc[(met['RNET']<=0) | (met['GPP']<0)] = 0
    # fixed parameters to be refined.
    SRparas = SoilRoot(3.5e-4,-0.00696,2.71,3.3,0.395,0.395,0.3,3.5,5,5,57.12,9.75) 
    met['GA'],met['GA_U'],met['GSOIL'] = cal_ga(met,SRparas)
    met = met[met['TIMESTAMP_START']>lai['TIMESTAMP_START'][0]]
    return met.reset_index(),SRparas

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

def FindTargetYear(met,lqt,hqt,printflag):
    # Choose the optimal time period for parameter retrieval based on one of the following:
    # (1) A year with moderate precipitation qt \in (0.25,0.75)
    # (2) Within (1), the year with the highest measured data coverate
    # Try to retrieve from wet, median and dry years and see the difference
    yearrange = np.arange(met['TIMESTAMP_START'].min().year,met['TIMESTAMP_START'].max().year+1)
    DataQuality = []
    for y in yearrange:
        st = datetime(y,10,1,0,0)
        et = datetime(y+1,9,30,23,30)
        if st>=met['TIMESTAMP_START'].min() and et<=met['TIMESTAMP_START'].max():
            metyear = met.loc[(met['TIMESTAMP_START'] >= st) & (met['TIMESTAMP_START'] <= et)]
            QC = int(sum(np.all((metyear[['LE_F_MDS_QC','SW_IN_F_MDS_QC','SWC_F_MDS_AVG_QC']]<3),axis=1))/len(metyear)*100)
            QC_LAI = int(sum(metyear['LAI_QC']==0)/len(metyear)*100)
            MAP = metyear['P'].sum().astype(int)
        else:
            QC = np.nan
            QC_LAI = np.nan
            MAP = np.nan
        DataQuality = np.concatenate([DataQuality,[int(y),QC,QC_LAI,MAP]])
    DataQuality = pd.DataFrame(np.reshape(DataQuality,[len(yearrange),4]),columns=['Year','QC','QC_LAI','MAP'])
    # choose the best year for parameter retrieval
    targetyear = DataQuality
    qctrd = 1/2
    # criteria 1: more than qctrd data coverage of LAI  and MET
    if len(targetyear)>1 and len(targetyear.loc[(targetyear['QC_LAI']>qctrd) & (targetyear['QC']>qctrd)])>0:
        targetyear = targetyear.loc[(targetyear['QC_LAI']>qctrd) & (targetyear['QC']>qctrd)]
    # criteria 2: moderate precipitation
    targetyear = targetyear.loc[(targetyear['MAP']>DataQuality['MAP'].quantile(lqt)) & (targetyear['MAP']<DataQuality['MAP'].quantile(hqt))]
    # criteria 3: best met data coverage
    targetyear = targetyear['Year'].loc[targetyear['QC']==targetyear['QC'].max()].values[0].astype(int)
    if printflag>0:
        print(DataQuality)   
        print('Target year: '+str(targetyear))
    return targetyear


def cal_ga(df,SRparas): # (measurment height (m), canopy height (m))
    ch,z = (SRparas.ch,SRparas.z)
    zd = 0.67*ch # displacement height
    zm = 0.1*ch # Campbell and Norman, 1998
    zh = 0.2*zm
    eta = np.array(-karman*g*z*df['H']/(rhohat*Cpmol*df['TEMP']*df['USTAR']**3))
    psih = np.zeros(eta.shape)
    psih[eta>=0] = 6*np.log(1+eta[eta>=0]) # stable atmosphere
    psim = np.copy(psih)
    psih[eta<0] = -2*np.log((1+(1-16*eta[eta<0])**0.5)/2) # unstable atmosphere
    psim = 0.6*psih
    # Campbell and Norman, 1998
    ga = np.array(karman**2*rhohat*df['WS']/((np.log((z-zd)/zm)+psim)*(np.log((z-zd)/zh)+psih))) # mol/m2/s
    ga[ga<=0] = np.percentile(ga[ga<=0],0.05)
    a = 3; z0h = 0.005
    # Ivanov, Ph.D. thesis, consistant with CLM
    ga_under = 1/(ch/(a*karman*df['USTAR']*(ch-zd))*(np.exp(a*(1-z0h/ch))-np.exp(a*(1-(zh+zd)/ch)))) 
    beta_ew = (df['SOILM']/SRparas.n1-SRparas.sw1)/(SRparas.sfc1-SRparas.sw1)
    beta_ew[beta_ew>1] = 1; beta_ew[beta_ew<0] = 0
    gsoil = 1/np.exp(8.206-4.255*beta_ew)*rhohat
    return ga,ga_under,gsoil

def CombineObsv(df,warmup,nobsinaday):
    discard = dailyAvg(df['P'],nobsinaday)>10/nobsinaday; discard[:warmup] = True
    observed_ET = dailyAvg(df['ET'],nobsinaday)[~discard]
    observed_GPP = dailyAvg(df['GPP'],nobsinaday)[~discard]
    rescale_ET = [np.mean(observed_ET),np.std(observed_ET)] 
    rescale_GPP = [np.mean(observed_GPP),np.std(observed_GPP)] 
    
    observed_day_valid = np.concatenate([(observed_ET-rescale_ET[0])/rescale_ET[1],
                                         (observed_GPP-rescale_GPP[0])/rescale_GPP[1]])
    return observed_day_valid, discard, rescale_ET, rescale_GPP

def dailyAvg(data,windowsize):
    data = np.array(data)
    data = data[0:windowsize*int(len(data)/windowsize)]
    return np.nanmean(np.reshape(data,[int(len(data)/windowsize),windowsize]),axis=1)

def f_PM(df,SRparas,gsref,sstar,m):
    fswc = np.array((df['SOILM']/SRparas.n1-SRparas.sw1)/(sstar-SRparas.sw1))
    fswc[fswc<0] = 0; fswc[fswc>1] = 1
    df['VPD'][df['VPD']<1e-3/101.325] = 1e-3/101.325
    gs = gsref*(1-m*np.log(df['VPD']*101.325))*fswc
    
    # partition energy into two parts, for ground and leaves respectively
#    df['RNET'][df['RNET']<0] = 0
    RNg = df['RNET']*np.exp(-df['LAI']*df['Vegk'])
    RNl = df['RNET']-RNg
    
    # ground ET
    ggh = np.array(df['GA_U'])
    ggv = np.array(1/(1/ggh+1/df['GSOIL']))
    Eg = PenmanMonteith(df['TEMP'],RNg,df['VPD'],ggv,ggh) # mol/m2/s
    
    # leaf ET
    glh = np.array(df['GA'])
    glv = 1/(1/glh+1/(gs*df['LAI']))
    El = PenmanMonteith(df['TEMP'],RNl,df['VPD'],glv,glh) # mol/m2/s
    return np.array(Eg+El)

T2ES  = lambda x: 0.6108*np.exp(17.27*(x-UNIT_3)/(x-UNIT_3+237.3))# saturated water pressure, kPa

def PenmanMonteith(temp,rnet,vpd,gv,gh): # vpd in mol/mol
    Delta = 4098*T2ES(temp)*1e3/(237.3+temp-UNIT_3)**2 # Pa/K
    E = (Delta*rnet+p0*Cpmol*gh*vpd)/(Delta*lambdamol+p0*Cpmol*gh/gv)
    return E*(rnet>0)


def f_PM_hydraulics_minimal(df,SRparas,XLparas,Init):
    RNg = df['RNET']*np.exp(-df['LAI']*df['Vegk'])
    RNl = df['RNET']-RNg
    
    ggh = np.array(df['GA_U'])
    ggv = np.array(1/(1/ggh+1/df['GSOIL']))
    Eg = PenmanMonteith(df['TEMP'],RNg,df['VPD'],ggv,ggh) # mol/m2/s

    PSIL = np.zeros([len(df),])+Init.psil
    El = np.zeros([len(df),])
    An = np.zeros([len(df),])
    for t in range(1,len(df)):
        Clm = Climate(df['TEMP'][t],df['VPD'][t],RNl[t],df['LAI'][t])
        El[t],PSIL[t],psir,s2,An[t] = calSPAC_minimal(Clm,df['GA'][t],SRparas,XLparas,Init)
        if t>47:psil_avg = np.mean(PSIL[t-47:t+1])
        else: psil_avg = Init.psil_avg
        Init = InitialState(psir,Init.psix,PSIL[t],psil_avg,df['SOILM'][t]/SRparas.n1,s2)
    return El+Eg, An

def calSPAC_minimal(Clm,glh,SRparas,XLparas,Init):
    ll = MWU(XLparas.lww,XLparas.b0,ca,Init.psil_avg)
    gs, An, H = f_carbon(Clm,ll) # mol H2O /m2/s
    gs = gs*a0*Clm.lai
    glv = 1/(1/glh+1/gs)
    E = PenmanMonteith(Clm.temp,Clm.rnet,Clm.vpd,glv,glh) # mol/m2/s
    Ems = E*18*1e-6 # m/s
    
    psis,gsr,s2 = soilhydro(Init,SRparas)
    gp = VulnerabilityCurve(XLparas.gpmax,XLparas.p50,XLparas.aa,Init.psil)
    # M3
    gsrp = 1/(1/gsr+1/gp)
    psil = psis-Ems/gsrp
    psir = psil+Ems/gp
    
    xx = np.arange(-10,-0.1,0.1)
    Esupply = (psis-xx)/(1/gsr+1/VulnerabilityCurve(XLparas.gpmax,XLparas.p50,XLparas.aa,xx))
    psil_min = xx[Esupply==max(Esupply)]
    psil = max(psil,psil_min)
#    Esupply = min(Ems,max(Esupply))
    return E,psil,psir,s2,An

def VulnerabilityCurve(gpmax,p50,aa,psil):
    return gpmax/(1+(psil/p50)**aa)

def MWU(lww,b0,ca,psil):
    return ca/ca0*lww*np.exp(b0*psil)

def f_carbon(Clm,ll):
#    ll = np.array([ll])
    T,RNET,VPD = (Clm.temp,Clm.rnet,Clm.vpd)
    
    hc = 2e-25 # Planck constant times light speed, J*s times m/s
    wavelen = 500e-9 # wavelength of light, m
    EE = hc/wavelen # energy of photon, J
    NA = 6.02e23 # Avogadro's constant, /mol
    PAR = RNET/(EE*NA)*1e6 # absorbed photon irradiance, umol photons /m2/s, PAR
    
    Vcmax = koptv*Hdv*np.exp(Hav*(T-Toptv)/T/R/Toptv)/(Hdv-Hav*(1-np.exp(Hav*(T-Toptv)/T/R/Toptv))) # umol/m2/s
    Jmax = koptj*Hdj*np.exp(Haj*(T-Toptj)/T/R/Toptj)/(Hdj-Haj*(1-np.exp(Haj*(T-Toptj)/T/R/Toptj)))
    TC = T-UNIT_3 # C
    Kc = 300*np.exp(0.074*(TC-25)) # umol/mol
    Ko = 300*np.exp(0.015*(TC-25)) # mmol/mol
    cp = 36.9+1.18*(TC-25)+0.036*(TC-25)**2
    J = (kai2*PAR+Jmax-np.sqrt((kai2*PAR+Jmax)**2-4*kai1*kai2*PAR*Jmax))/2/kai1 # umol electrons /m2/s
    Rd = 0.015*Vcmax

    a1 = J/4;a2 = 2*cp # Rubisco limited photosynthesis
    tmpsqrt = a0*VPD*ll*a1**2*(ca-cp)*(a2+cp)*(a2+ca-2*a0*VPD*ll)**2*(a2+ca-a0*VPD*ll)
    if tmpsqrt>0:
        gc1 = -a1*(a2-ca+2*cp)/(a2+ca)**2+np.sqrt(tmpsqrt)/(a0*VPD*ll*(a2+ca)**2*(a2+ca-a0*VPD*ll))
        A = -gc1
        B = gc1*ca-a2*gc1-a1+Rd
        C = ca*a2*gc1+a1*cp+a2*Rd
        ci1 = (-B-np.sqrt(B**2-4*A*C))/(2*A)
        An1 = gc1*(ca-ci1)
        if np.isnan(An1) or An1<0 or gc1<0: gc1, ci1, An1 = (0,0,0)
    else:
        gc1, ci1, An1 = (0,0,0)
        
    a1 = Vcmax;a2 = Kc*(1+Coa/Ko) # RuBP limited photosynthesis
    tmpsqrt = a0*VPD*ll*a1**2*(ca-cp)*(a2+cp)*(a2+ca-2*a0*VPD*ll)**2*(a2+ca-a0*VPD*ll)
    if tmpsqrt>0:
        gc2 = -a1*(a2-ca+2*cp)/(a2+ca)**2+np.sqrt(tmpsqrt)/(a0*VPD*ll*(a2+ca)**2*(a2+ca-a0*VPD*ll))
        A = -gc2
        B = gc2*ca-a2*gc2-a1+Rd
        C = ca*a2*gc2+a1*cp+a2*Rd
        ci2 = (-B-np.sqrt(B**2-4*A*C))/(2*A)
        An2 = gc2*(ca-ci2)
        if np.isnan(An2) or An2<0 or gc2<0: gc2, ci2, An2 = (0,0,0)
    else:
        gc2, ci2, An2 = (0,0,0)
        
    flag = (An1<=An2)
    An = min(An1,An2)
    gc = gc1*flag+gc2*(1-flag)
    ci = ci1*flag+ci2*(1-flag)
    H = gc*(ca-ci)-a0*gc*VPD*ll
    return max(gc*(H>=0),1e-5),An, H

def f_psis(s,psat,b): return psat*np.power(s,-b);

def soilhydro(Init,SRparas):
    K1 = SRparas.ksat1*np.power(Init.s1,SRparas.b1*2+3)
    K2 = SRparas.ksat2*np.power(Init.s2,SRparas.b2*2+3)
    K = (SRparas.Zr1+SRparas.Zr2)/(SRparas.Zr1/K1+SRparas.Zr2/K2)
    P1 = f_psis(Init.s1,SRparas.psat1,SRparas.b1)+(SRparas.Zr1+SRparas.Zr2)/2*rhow*g/UNIT_2
    P2 = f_psis(Init.s2,SRparas.psat2,SRparas.b2)
    gsr1 = K1*np.sqrt(SRparas.RAI1)/(rhow*g*SRparas.Zr1*np.pi)*UNIT_2
    gsr2 = K2*np.sqrt(SRparas.RAI2)/(rhow*g*SRparas.Zr2*np.pi)*UNIT_2
    gsr = gsr1+gsr2
    psis = (gsr1*P1+gsr2*P2)/gsr
    E2 = gsr2*(P2-Init.psir)
    L = K*(P1-P2)/(rhow*g*(SRparas.Zr1+SRparas.Zr2)/2/UNIT_2) # m/s
    s2 = min(max(Init.s2+(L-E2)/SRparas.n2/SRparas.Zr2*dt,SRparas.sw2),1)
    return psis,gsr,s2   
