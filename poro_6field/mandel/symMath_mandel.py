#!/usr/bin/env python

# Symbolic mathematics script using sympy

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import cm
from matplotlib.colors import LightSource, Normalize


sp.init_printing(use_unicode=True)

# Positional Variables, Cartesian
x, y, z, t = sp.symbols('x, y, z, t')

# Positional Variables, Spherical
R, theta, phi = sp.symbols('R, theta, phi')

# Incremental Variables
i, j, k, n = sp.symbols('i, j, k, n', integer=True)

# Coefficients
F, K_u, nu, nu_u, G, a, alpha_n, c_f, alpha, M = sp.symbols('F, K_u, nu, nu_u, G, a, alpha_n, c_f, alpha, M')

# =====================================
# Mandel Solution for Volumetric Strain

# Analytical solutions for displacement
# For isotropic conditions:
K_d = K_u - alpha*alpha*M
# nu = (3*K_d - 2*G)/(2*(3*K_d + G))
# nu_u = (3*K_u - 2*G)/(2*(3*K_u + G))
B = (3 * (nu_u - nu) )/(alpha*(1-2*nu)*(1+nu_u))

A_1 = 3 / (B * (1 + nu_u))
A_2 = (alpha * (1 - 2*nu))/(1 - nu)

P_0 = ( 1 / (3*a) ) * B*(1 + nu_u) * F

A_s = sp.Sum( ( (sp.sin(alpha_n)*sp.cos(alpha_n)) / (alpha_n - sp.sin(alpha_n)*sp.cos(alpha_n)) ) * sp.exp( -1*(alpha_n**2 * c_f *t)/(a**2)), (n, 1, sp.oo) )
B_s = sp.Sum( (sp.cos(alpha_n) / (alpha_n - sp.sin(alpha_n)*sp.cos(alpha_n)) ) * sp.sin((alpha_n*x)/a) * sp.exp( -1*(alpha_n**2*c_f*t)/(a**2) ), (n,1,sp.oo))
P_s = sp.Sum( ( (sp.sin(alpha_n)) / (alpha_n - sp.sin(alpha_n)*sp.cos(alpha_n)) ) * ( sp.cos( (alpha_n * x)/a ) - sp.cos(alpha_n) ) * sp.exp( -1*(alpha_n**2 * c_f *t)/(a**2)), (n, 1, sp.oo) )

u_x = ( ( (F*nu)/(2*G*a) ) - ( (F*nu_u)/(G*a) ) * A_s )*x + (F/G) * B_s

u_y = ( (-1*(F*(1-nu))/(2*G*a)) + ( (F*(1-nu_u))/(G*a) ) * A_s )*y

p = 2*P_0 * P_s

# Analytical solution for volumetric strain
eps = sp.diff(u_x,x) + sp.diff(u_y,y)

# Time derivative formulations
u_x_t = sp.diff(u_x,t)
u_y_t = sp.diff(u_y,t)
p_t = sp.diff(p,t)
eps_t = sp.diff(eps,t)


# Analytical solution for sigma_zz (isotropic condition)

E = 2.0*G*(1 + nu)

E_x = E
E_z = E
alpha_1 = alpha
alpha_3 = alpha
nu_zx = nu
nu_yx = nu
G_xz = G

M_11 = ( E_x*(E_z - E_x * nu_zx**2) ) / ( (1 + nu_yx)*(E_z - E_z*nu_yx - 2*E_x * nu_zx**2) )
M_12 = ( E_x*(nu_yx*E_z + E_x * nu_zx**2) ) / ( (1 + nu_yx)*(E_z - E_z*nu_yx - 2*E_x * nu_zx**2) )
M_13 = ( E_x*E_z*nu_zx ) / ( E_z - E_z*nu_yx - 2*E_x * nu_zx**2 )
M_33 = ( E_z**2 * (1 - nu_yx) ) / (E_z - E_z*nu_yx - 2*E_x * nu_zx**2) 
M_55 = G_xz

#A_1 = (alpha_1**2 * M_33 - 2*alpha_1*alpha_3*M_13 + alpha_3**2 * M_11)/(alpha_3*M_11 - alpha_1*M_13) + (M_11*M_33 - M_13**2)/(M * (alpha_3*M_11 - alpha_1*M_13) )
#A_2 = (alpha_3*M_11 - alpha_1*M_13) / M_11



sigma_zz = -F/a - ( (2*F*A_2)/(a*A_1) ) * sp.Sum( ( sp.sin(alpha_n)/(alpha_n - sp.sin(alpha_n)*sp.cos(alpha_n) ) ) * sp.cos( (alpha_n*x)/a ) * sp.exp( -(alpha_n**2 * c_f*t)/(a**2) ), (n,1,sp.oo)) \
          + ((2*F)/a) * sp.Sum( ( (sp.sin(alpha_n)*sp.cos(alpha_n))/(alpha_n - sp.sin(alpha_n)*sp.cos(alpha_n)) ) * sp.exp( -(alpha_n**2*c_f*t)/(a**2) ), (n,1,sp.oo) )
          

# ==============================================================================
# Numeric Sandbox
G_num = 3.0
K_fl_num = 8.0
K_sg_num = 10.0
K_d_num = 4.0
alpha_num = 0.6
phi_num = 0.1
k_num = 1.5
mu_f_num = 1.0
P_0_num = 1.0

#G_num = 18e9 # Pa
#K_fl_num = 8.0
#K_sg_num = 10.0
#K_d_num = 4.0
#alpha_num = 0.733
#phi_num = 0.1
#k_num = 1e7 * 0.987e-12 # darcy ++> m**2
#mu_f_num = 0.001 # Pa*s
#P_0_num = 1e6 # N/m

ndim = 2

xmin = 0.0
ymin = 0.0
xmax = 10.0
ymax = 1.0
tmin = 0.0
tmax = 40.0

t_0 = 0.0
dt = 0.1


M_num    = 1.0 / ( phi_num / K_fl_num + (alpha_num - phi_num) /K_sg_num)
K_u_num = K_d_num + alpha_num*alpha_num*M_num
nu_num = (3*K_d_num - 2*G_num) / (2*(3*K_d_num + G_num))
nu_u_num = (3*K_u_num - 2*G_num) / (2*(3*K_u_num + G_num))
kappa_num = k_num / mu_f_num
a_num = (xmax - xmin)
b_num = (ymax - ymin)
c_num = ( (2*kappa_num*G_num) * (1 - nu_num) * (nu_u_num - nu_num) ) / (alpha_num*alpha_num * (1 - 2*nu_num) * (1 - nu_num) )
B_num = (3 * (nu_u_num - nu_num) )/(alpha_num*(1-2*nu_num)*(1+nu_u_num))
A_1_num = 3 / (B_num * (1 + nu_u_num))
A_2_num = (alpha_num * (1 - 2*nu_num))/(1 - nu_num)

# Load grid
filename = 'mesh_quad.exo'
quad = Dataset('mesh_quad.exo','r')  
X = np.array(quad.variables['coordx'][:])
Y = np.array(quad.variables['coordy'][:])
xy = np.array([X[:], Y[:]]).T
connect = quad.variables['connect1']

# Generate zeros
NITER = 2000
EPS = 1e-5
zeroArray = np.zeros(NITER)
for i in np.arange(1,NITER+1):
  a1 = (i - 1) * np.pi * np.pi / 4 + EPS
  a2 = a1 + np.pi / 2
  for j in np.arange(0,NITER):
    y1 = np.tan(a1) - (1 - nu_num) / (nu_u_num - nu_num)*a1
    y2 = np.tan(a2) - (1 - nu_num) / (nu_u_num - nu_num)*a2
    am = (a1 + a2) / 2
    ym = np.tan(am) - (1 - nu_num) / (nu_u_num - nu_num)*am
    if ym*y1 > 0:
      a1 = am
    else:
      a2 = am
    if np.abs(y2) < EPS:
      am = a2
  zeroArray[i-1] = am


# Generate sigma_zz distribution
F_num = P_0_num
#t_pos = np.arange(0.0, 0.0057333334, 0.0028666667) # sec #np.linspace(tmin, tmax)

t_pos = np.arange(t_0, tmax + dt, dt) # sec #np.linspace(tmin, tmax)
x_pos = X.reshape([1,X.size]) #np.linspace(xmin, xmax)
y_pos = Y.reshape([1,Y.size]) #np.linspace(ymin, ymax)

sigma_zz_num = np.zeros([x_pos.size,t_pos.size])
pressure = np.zeros([x_pos.size,t_pos.size])
displacement = np.zeros([x_pos.size,ndim,t_pos.size])
velocity = np.zeros([x_pos.size,ndim,t_pos.size])

t_count = 0
for t_num in t_pos:
  x_count = 0
#  for x_num in x_pos:
  sigma_zz_A_sum = 0
  sigma_zz_B_sum = 0
  U_A_x = 0
  U_B_x = 0  
  U_A_t = 0
  U_B_t = 0
  P_it = 0
#  for i in np.arange(1,NITER):
  alpha_n_num = zeroArray.reshape([zeroArray.size,1])
  
  sigma_zz_A_sum += ( np.sin(alpha_n_num) / (alpha_n_num - np.sin(alpha_n_num)*np.cos(alpha_n_num)) ) * np.cos( (alpha_n_num*x_pos) / a_num ) * np.exp( -1.0*(alpha_n_num*alpha_n_num*c_num*t_num)/(a_num*a_num))   
  sigma_zz_B_sum += ( (np.sin(alpha_n_num)*np.cos(alpha_n_num)) / (alpha_n_num - np.sin(alpha_n_num)*np.cos(alpha_n_num)) ) * np.exp( -1.0*(alpha_n_num*alpha_n_num*c_num*t_num)/(a_num*a_num))
  
  U_A_x += (np.sin(alpha_n_num)*np.cos(alpha_n_num) / (alpha_n_num - np.sin(alpha_n_num)*np.cos(alpha_n_num)))*np.exp(-1.0*(alpha_n_num*alpha_n_num*c_num*t_num)/(a_num*a_num))
  U_B_x += (np.cos(alpha_n_num) / (alpha_n_num - np.sin(alpha_n_num)*np.cos(alpha_n_num))) * np.sin( (alpha_n_num*x_pos)/a_num) * np.exp(-1.0*(alpha_n_num*alpha_n_num*c_num*t_num)/(a_num*a_num))

  U_A_t += ( (-1*alpha_n_num*alpha_n_num*c_num*np.exp(-1.0*(alpha_n_num*alpha_n_num*c_num*t_num)/(a_num*a_num))*np.sin((alpha_n_num*x_pos)/a_num)*np.cos(alpha_n_num))/(a_num*a_num*(alpha_n_num*np.sin(alpha_n_num)*np.cos(alpha_n_num))) )
  U_B_t += ( (-1*alpha_n_num*alpha_n_num*c_num*np.exp(-1.0*(alpha_n_num*alpha_n_num*c_num*t_num)/(a_num*a_num))*np.sin(alpha_n_num)*np.cos(alpha_n_num))/(a_num*a_num*(alpha_n_num*np.sin(alpha_n_num)*np.cos(alpha_n_num))) )

  P_it += ( (np.sin(alpha_n_num))/(alpha_n_num - np.sin(alpha_n_num)*np.cos(alpha_n_num)) ) * (np.cos( (alpha_n_num*x_pos)/a_num) - np.cos(alpha_n_num))*np.exp( -1.0*(alpha_n_num*alpha_n_num*c_num*t_num)/(a_num*a_num)) 

  sigma_zz_A_sum = sigma_zz_A_sum.sum(axis=0)
  sigma_zz_B_sum = sigma_zz_B_sum.sum(axis=0)
  U_A_x = U_A_x.sum(axis=0)
  U_B_x = U_B_x.sum(axis=0)
  U_A_t = U_A_t.sum(axis=0)    
  U_B_t = U_B_t.sum(axis=0)
  P_it = P_it.sum(axis=0)

  sigma_zz_num[:, t_count] = -(F_num/a_num) - ((2*F_num) / a_num) * (A_2_num / A_1_num) * sigma_zz_A_sum + ((2*F_num) / a_num) * sigma_zz_B_sum
  pressure[:, t_count] = ( (2.0*F_num*B_num*(1.0+nu_u_num))/(3.*a_num) ) * P_it
  displacement[:, 0, t_count] = ((F_num*nu_num)/(2.0*G_num*a_num) - (F_num*nu_u_num)/(G_num*a_num) * U_A_x ) * x_pos + F_num/G_num * U_B_x
  displacement[:, 1, t_count] = (-1*(F_num*(1.0-nu_num))/(2*G_num*a_num) + (F_num*(1-nu_u_num))/(G_num*a_num) * U_A_x)*y_pos
  velocity[:,0,t_count] = (F_num*U_A_t)/G_num - (F_num*nu_u_num*x_pos * U_B_t)/(G_num*a_num)
  velocity[:,1,t_count] = (F_num*y_pos*(1 - nu_u_num)*U_B_t)/(G_num*a_num)
#    x_count += 1
  t_count += 1

# ==============================================================================
# Generate Analytical Solution Plots
# ==============================================================================
#y_pos = np.where(Y == ymax)[0]
#x_pos = np.where(X == xmax)[0]
#x_fig = X[y_pos]
#y_fig = Y[x_pos]
#x_fig, tx_fig = np.meshgrid(x_fig, t_pos)
#y_fig, ty_fig = np.meshgrid(y_fig, t_pos)


# Pressure
#fig = plt.figure()
#fig.set_size_inches(15,10)
#ax = fig.gca(projection='3d')

#surf_pressure = ax.plot_surface(x_fig, tx_fig, pressure[y_pos,:].T, cmap=cm.coolwarm)
#fig.colorbar(surf_pressure, shrink=0.5, aspect=5)




#ax.plot_surface(x_fig, tx_fig, displacement[y_pos,0,:].T, cmap=cm.coolwarm)
#ax.plot_surface(y_fig, ty_fig, displacement[x_pos,1,:].T, cmap=cm.coolwarm)
#plt.show()


# Generate displacement at Z+ 

#ypos_disp = np.zeros([t_pos.size,2])
#ypos_disp[:,0] = t_pos[:]
#ypos_disp[:,1] = displacement[y_pos[0],1,:]

          
          

