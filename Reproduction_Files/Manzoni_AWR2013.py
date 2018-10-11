import numpy as np
import matplotlib.pyplot as plt
import sympy
import scipy
import pdb
#from scipy.integrate import odeint


#x0=0.8
#ca=350
#k=0.05
#T=20
#y=0.001
#n=0.5
#Zr=0.3
#D=0.015
#LAI=4
#M_w=0.018
#rho_w=1000
#a=1.6
#T_day=12*3600






### First Case c=1

x0=sympy.Symbol('x0') # initial condition of the soil moisture
ca=sympy.Symbol('ca') # carbon in the atmosphere, ca=350 micro_mol.mol^-1
k=sympy.Symbol('k') # Carboxylation efficiency k=0.05 mol.m^-2.s^-1
T=sympy.Symbol('T') # period of the dry, T=20 days
y=sympy.Symbol('y') # Parameter of the water loss function 1 mm.d^-1
n=sympy.Symbol('n') # Soil porisity, n=0.5 m^3.m^-3
Zr=sympy.Symbol('Zr') # Rooting depth, Zr=0.3m
D=sympy.Symbol('D') # Vapor pressure deficit, D=0.015 mol.mol^-1
LAI=sympy.Symbol('LAI') # Leaf Area Index
M_w=sympy.Symbol('M_w') # Molar mass of water M_w=0.018 kg.mol^-1
rho_w=sympy.Symbol('rho_w') # Densiy of liquid water rho_w=1000 kg.m^-3
a=sympy.Symbol('a') # Ratio of water vapor to CO2 diffusivity a=1.6
v=sympy.Symbol('v') # Unit Conversion factor m^3.s.mol^-1.d^-1
T_day=sympy.Symbol('T_day') # Day length in seconds
c=sympy.Symbol('c') # uncottrolled losses
alpha=sympy.Symbol('alpha') # Combination of parameters
beta=sympy.Symbol('beta') # Combination of parameters
c=1
v=(LAI*T_day*M_w)/rho_w
alpha=v*a/(n*Zr)
beta=y/(n*Zr)

# Variables
lam0=sympy.Symbol('lam0') # initial condition of lambda
t=sympy.Symbol('t')
g= sympy.Function('g')(t)
lam= sympy.Function('lam')(t)
x= sympy.Function('x')(t)
A= sympy.Function('A')
f= sympy.Function('f')
H= sympy.Function('H')

#Equations
A=(g*ca*k)/(g+k)
f=-alpha*D*g-beta*x**c
H=A+lam*f


## First equation
dHdg=sympy.diff(H,g)
#print('\nthis is del H by del g:')
#print(dHdg)
g_t=sympy.solve(dHdg,g)
#print('\nthis is g in function of lambda:')    
#print(g_t)
dHdx=sympy.diff(H,x)
#print('\nthis is del H by del x:')
#print(dHdx)
ode=sympy.Eq(sympy.diff(lam,t),-dHdx) # Setting the equation of lambda with dH/dx
#print('\nthis is the equation dlam/dt=-dH/dx:')
#print(ode)
lam_t=sympy.dsolve(ode,lam) # Solving the equation lambda(t)
#print('\nthis is the equation of lambda:')
#print(lam_t)
lam_0=lam_t.subs([(t,0)])
lam_0=lam_0.subs([(lam_0.lhs,lam0)])
#print('\nThis is the equation of lambda at t=0:')
#print(lam_0)
lam_t=lam_t.subs([(lam_0.rhs,lam_0.lhs)])
#print('\nThis is the equation of lambda in function of the initial condition of lambda:')
#print(lam_t)
g_t=g_t[0].subs([(lam,lam_t.rhs)])
#print('\nthis is g in function of lambda:')
#print(g_t)
ode1=sympy.Eq(sympy.diff(x,t),f)
ode1=ode1.subs([(g,g_t)])
#print('\nthis is the equation dx/dt=f')
#print(ode1)
x=sympy.dsolve(ode1,x)
#print('\nthis is the equation of x:')
#print(x)
constants=sympy.solve(x.rhs.subs(t,0)-x0)
#print('\n')
#print(constants)
x=x.subs(constants[0])
#print('\nthis is the equation of soil moisture in function of the initial condition of x:')
#print(x)
x_T=x.subs([(t,T)])
x_T=x_T.subs([(x_T.lhs,0)])
#print('\nthis is the equation of x at t=T:')
#print(x_T)
nb=sympy.solve(x_T.rhs,lam0)
#print(nb)
lam_t=lam_t.subs([(lam0,nb[0])])
#print('\nthis is the equation of lambda')
#print(lam_t.rhs)
g_t=g_t.subs([(lam0,nb[0])])
#print('\nthis is the equation of g')
#print(g_t)
pdb.set_trace()
something=g_t.subs([(a,1.6),(T_day,12*3600),(rho_w,1000),(M_w,0.018),(LAI,4),
                    (D,0.015),(Zr,0.3),(n,0.5),(y,0.001),(T,20),(k,0.05),
                    (ca,350),(x0,0.8)])
#print('\nthis is the equation of g when c=1')
#print(something)
something1=lam_t.rhs.subs([(a,1.6),(T_day,12*3600),(rho_w,1000),(M_w,0.018),
                           (LAI,4),(D,0.015),(Zr,0.3),(n,0.5),(y,0.001),(T,20),
                           (k,0.05),(ca,350),(x0,0.8)])
#print('\nthis is the equation of lambda when c=1')
#print(something1)


### Second Case c=0
c=0
lam0=sympy.Symbol('lam0') # initial condition of lambda

# Variables
lam0=sympy.Symbol('lam0') # initial condition of lambda
t=sympy.Symbol('t')
g= sympy.Function('g')(t)
lam= sympy.Function('lam')(t)
x= sympy.Function('x')(t)
A= sympy.Function('A')
f= sympy.Function('f')
H= sympy.Function('H')

#Equations
A=(g*ca*k)/(g+k)
f=-alpha*D*g-beta*x**c
H=A+lam*f

## First equation
dHdg=sympy.diff(H,g)
#print('\nthis is del H by del g:')
#print(dHdg)
g_t=sympy.solve(dHdg,g)
#print('\nthis is g in function of lambda:')    
#print(g_t)
dHdx=sympy.diff(H,x)
#print('\nthis is del H by del x:')
#print(dHdx)
ode=sympy.Eq(sympy.diff(lam,t),-dHdx) # Setting the equation of lambda with dH/dx
#print('\nthis is the equation dlam/dt=-dH/dx:')
#print(ode)
lam_t=sympy.dsolve(ode,lam) # Solving the equation lambda(t)
#print('\nthis is the equation of lambda:')
#print(lam_t)
lam_0=lam_t.subs([(t,0)])
lam_0=lam_0.subs([(lam_0.lhs,lam0)])
#print('\nThis is the equation of lambda at t=0:')
#print(lam_0)
lam_t=lam_t.subs([(lam_0.rhs,lam_0.lhs)])
#print('\nThis is the equation of lambda in function of the initial condition of lambda:')
#print(lam_t)
g_t=g_t[0].subs([(lam,lam_t.rhs)])
#print('\nthis is g in function of lambda:')
#print(g_t)
ode1=sympy.Eq(sympy.diff(x,t),f)
ode1=ode1.subs([(g,g_t)])
#print('\nthis is the equation dx/dt=f')
#print(ode1)
x=sympy.dsolve(ode1,x)
#print('\nthis is the equation of x:')
#print(x)
constants=sympy.solve(x.rhs.subs(t,0)-x0)
#print('\n')
#print(constants)
x=x.subs(constants[0])
#print('\nthis is the equation of soil moisture in function of the initial condition of x:')
#print(x)
x_T=x.subs([(t,T)])
x_T=x_T.subs([(x_T.lhs,0)])
#print('\nthis is the equation of x at t=T:')
#print(x_T)
nb=sympy.solve(x_T.rhs,lam0)
#print(nb)
lam_t=lam_t.subs([(lam0,nb[0])])
#print('\nthis is the equation of lambda')
#print(lam_t.rhs)
g_t=g_t.subs([(lam0,nb[0])])
#print('\nthis is the equation of g')
#print(g_t)
something2=g_t.subs([(a,1.6),(T_day,12*3600),(rho_w,1000),(M_w,0.018),(LAI,4),
                     (D,0.015),(Zr,0.3),(n,0.5),(y,0.001),(T,20),(k,0.05),
                     (ca,350),(x0,0.8)])
#print('\nthis is the equation of g when c=0')
#print(something2)
something3=lam_t.rhs.subs([(a,1.6),(T_day,12*3600),(rho_w,1000),(M_w,0.018),
                           (LAI,4),(D,0.015),(Zr,0.3),(n,0.5),(y,0.001),(T,20),
                           (k,0.05),(ca,350),(x0,0.8)])
#print('\nthis is the equation of lambda when c=0')
#print(something3)

print('\nthis is the equation of lambda when c=0')
print(something3)
print('\nthis is the equation of g when c=0')
print(something2)
print('\nthis is the equation of lambda when c=1')
print(something1)
print('\nthis is the equation of g when c=1')
print(something)
p1=sympy.plot(something3,something1,(t,0,20),xlabel='time in days',ylabel='lambda')
p2=sympy.plot(something2,something,(t,0,20),xlabel='time in days',ylabel='Conductance')
