#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import h5py

# ==============================================================================
# Computational Values
ITERATIONS = 2000
EPS = 1e-5

# ==============================================================================
# Physical Values

G = 3.0
rho_s = 2500
rho_f = 1000
K_fl = 8.0
K_sg = 10.0
K_d = 4.0
alpha = 0.6
phi = 0.1
k = 1.5
mu_f = 1.0
F = 1.0
ndim = 2

zmax = 1.0
zmin = 0.0
ymax = 1.0
ymin = 0.0
xmax = 10.0
xmin = 0.0
P_0 = 1.0

M    = 1.0 / ( phi / K_fl + (alpha - phi) /K_sg)
K_u = K_d + alpha*alpha*M
nu = (3*K_d - 2*G) / (2*(3*K_d + G))
nu_u = (3*K_u - 2*G) / (2*(3*K_u + G))
kappa = k / mu_f
a = (xmax - xmin)
b = (ymax - ymin)
#c = ( (2*kappa*G) * (1 - nu) * (nu_u - nu) ) / (alpha*alpha * (1 - 2*nu) * (1 - nu) )
B_alt = (3 * (nu_u - nu) )/(alpha*(1-2*nu)*(1+nu_u))
B = (alpha*M) / (K_d + alpha*alpha*M) #
A_1 = 3 / (B * (1 + nu_u))
A_2 = (alpha * (1 - 2*nu))/(1 - nu)

E = 2.0*G*(1.0 + nu)

E_x = E
E_z = E
alpha_1 = alpha
alpha_3 = alpha
nu_zx = nu
nu_yx = nu
G_xz = G
kappa_x = kappa

M_11 = ( E_x*(E_z - E_x * nu_zx**2) ) / ( (1 + nu_yx)*(E_z - E_z*nu_yx - 2*E_x * nu_zx**2) )
M_12 = ( E_x*(nu_yx*E_z + E_x * nu_zx**2) ) / ( (1 + nu_yx)*(E_z - E_z*nu_yx - 2*E_x * nu_zx**2) )
M_13 = ( E_x*E_z*nu_zx ) / ( E_z - E_z*nu_yx - 2*E_x * nu_zx**2 )
M_33 = ( E_z**2 * (1 - nu_yx) ) / (E_z - E_z*nu_yx - 2*E_x * nu_zx**2) 
M_55 = G_xz

A_1 = (alpha_1**2 * M_33 - 2*alpha_1*alpha_3*M_13 + alpha_3**2 * M_11)/(alpha_3*M_11 - alpha_1*M_13) + (M_11*M_33 - M_13**2)/(M * (alpha_3*M_11 - alpha_1*M_13) )
A_2 = (alpha_3*M_11 - alpha_1*M_13) / M_11

c_x = (kappa_x*M*M_11)/(M_11 + alpha_1*alpha_1*M)
B_prime = (alpha*M)/(K_d + alpha*alpha*M)

K_v = 2.*G*( (1.-nu) / (1.-2.*nu) )
c_f = kappa*M*( K_v / (K_v + alpha*alpha*M) )
c = kappa*M*( K_v / (K_v + alpha*alpha*M) )
# ==============================================================================
def mandelZeros():
    """
    Compute roots for analytical mandel problem solutions
    """
    zeroArray = np.zeros(ITERATIONS)

    for i in np.arange(1, ITERATIONS+1,1):
        a1 = (i - 1.0) * np.pi * np.pi / 4.0 + EPS
        a2 = a1 + np.pi / 2
        am = a1
        for j in np.arange(0, ITERATIONS,1):
            y1 = np.tan(a1) - ((1.0 - nu) / (nu_u - nu))*a1
            y2 = np.tan(a2) - ((1.0 - nu) / (nu_u - nu))*a2
            am = (a1 + a2) / 2.0
            ym = np.tan(am) - (1 - nu) / (nu_u - nu)*am
            if ((ym*y1) > 0):
                a1 = am
            else:
                a2 = am
            if (np.abs(y2) < EPS):
                am = a2
        zeroArray[i-1] = am
    return zeroArray

def displacement(locs,tsteps,zeroArray):
    """
    Compute displacement field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    displacement = np.zeros((ntpts, npts, dim), dtype=np.float64)
    x = locs[:,0]
    z = locs[:,1]
    t_track = 0
#    zeroArray = mandelZeros()

    for t in tsteps:
#        A_x = 0.0
#        B_x = 0.0
#        A_z = 0.0

        A_x = np.sum( (np.sin(zeroArray)*np.cos(zeroArray) / (zeroArray - np.sin(zeroArray)*np.cos(zeroArray))) * np.exp( -1.0*(zeroArray*zeroArray*c*t)/(a*a) ) )
        B_x = np.sum((np.cos(zeroArray) / (zeroArray - np.sin(zeroArray)*np.cos(zeroArray))) * np.sin( (zeroArray*x.reshape([x.size,1]))/a) \
            * np.exp(-1.0*(zeroArray*zeroArray*c*t)/(a*a)),axis=1)
        A_z = np.sum( (np.sin(zeroArray)*np.cos(zeroArray) / (zeroArray - np.sin(zeroArray)*np.cos(zeroArray)))*np.exp( -1.0*(zeroArray*zeroArray*c*t)/(a*a) ) )


        #for n in np.arange(1,ITERATIONS+1,1):
#            a_n = zeroArray[n-1]

            #A_x += (np.sin(a_n)*np.cos(a_n) / (a_n - np.sin(a_n)*np.cos(a_n)))*np.exp(-1.0*(a_n*a_n*c_x*t)/(a*a))
            #B_x += (np.cos(a_n) / (a_n - np.sin(a_n)*np.cos(a_n))) * np.sin( (a_n*x)/a) * np.exp(-1.0*(a_n*a_n*c_x*t)/(a*a))
            
            #A_z += (np.sin(a_n)*np.cos(a_n) / (a_n - np.sin(a_n)*np.cos(a_n)))*np.exp(-1.0*(a_n*a_n*c*t)/(a*a))

#            A_x += (np.sin(a_n)*np.cos(a_n) / (a_n - np.sin(a_n)*np.cos(a_n))) * np.exp( -1.0*(a_n*a_n*c_f*t)/(a*a) )
#            B_x += (np.cos(a_n) / (a_n - np.sin(a_n)*np.cos(a_n))) * np.sin( (a_n*x)/a ) * np.exp( -1.0*(a_n*a_n*c_f*t)/(a*a) )
            
#            A_z += (np.sin(a_n)*np.cos(a_n) / (a_n - np.sin(a_n)*np.cos(a_n)))*np.exp( -1.0*(a_n*a_n*c_f*t)/(a*a) )
        
#        print('t = ', t, ' A_x= ', A_x)
#        print('t = ', t, ' B_x= ', B_x)
#        print('t = ', t, ' A_z= ', A_z)                

        # Isotropic Formulation
#        displacement[t_track,:,0] = ( (F*nu)/(2.*G*a) - (F*nu_u)/(G*a) * A_x )*x + (F/G)*B_x
#        displacement[t_track,:,1] = ( -(F*(1.-nu))/(2.*G*a) + (F*(1-nu_u))/(G*a) * A_z )*z

        # Orthotropic Formulation
        displacement[t_track,:,0] = (F/a * M_13/(M_11*M_33 - M_13*M_13) - (2.*F)/a * (alpha_1*alpha_3*M + M_13)/(A_1*M*(alpha_3*M_11 - alpha_1*M_13))*A_x)*x + (2.*F*alpha_1)/(A_2*M_11) * B_x
        displacement[t_track,:,1] = (-F/a)*(M_11/(M_11*M_33 - M_13*M_13)) * (1.0 + 2.0*(A_2/A_1 - 1.0) * A_z)*z
        t_track += 1

    return displacement

def pressure(locs, tsteps, zeroArray):
    """
    Compute pressure field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    pressure = np.zeros((ntpts, npts), dtype=np.float64)
    x = locs[:,0]
    z = locs[:,1]
    t_track = 0
#    zeroArray = mandelZeros()
    p_0 = (1./(3.*a))*B*(1+nu_u)*F
    for t in tsteps:
        p = np.sum( np.sin(zeroArray)/(zeroArray - np.sin(zeroArray)*np.cos(zeroArray)) * (np.cos( (zeroArray*x.reshape([x.size,1]))/a) \
                    - np.cos(zeroArray))*np.exp(-1.0*(zeroArray*zeroArray*c*t)/(a*a)), axis=1 )
        
#        if t == 0.0:
#            pressure[t_track,:] = (1./(3.*a))*(B*(1.+nu_u))*F
#        else:
#            p = np.sum( (np.sin(zeroArray) / (zeroArray - np.sin(zeroArray)*np.cos(zeroArray))) \
#                        * (np.cos( (zeroArray*x.reshape([x.size,1])) / a) - np.cos(zeroArray)) * np.exp(-1.0*(zeroArray*zeroArray * c_x * t)/(a*a)), axis=1 )
#            p = 0.0
#            for n in np.arange(1, ITERATIONS+1,1):
#                x_n = zeroArray[n-1]
#                p += (np.sin(x_n) / (x_n - np.sin(x_n)*np.cos(x_n))) * (np.cos( (x_n*x) / a) - np.cos(x_n)) * np.exp(-1.0*(x_n*x_n * c_x * t)/(a*a))
#            pressure[t_track,:] = (2.*F)/(a*A_1)  * p

        # Isotropic Formulation
#        pressure[t_track,:] = 2.0*p_0 * p
        
        # Orthotropic Formulation
        pressure[t_track,:] = (2.*F)/(a*A_1)  * p        
        t_track += 1

    return pressure

def trace_strain(locs, tsteps, zeroArray):
    """
    Compute trace strain field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    trace_strain = np.zeros((ntpts, npts), dtype=np.float64)
    x = locs[:,0]
    z = locs[:,1]
    t_track = 0
#    zeroArray = mandelZeros()

    for t in tsteps:

        eps_A = 0.0
        eps_B = 0.0
        eps_C = 0.0

        for i in np.arange(1, ITERATIONS+1,1):
            x_n = zeroArray[i-1]
            eps_A += (x_n * np.exp( (-1.0*x_n*x_n*c*t)/(a*a)) * np.cos(x_n)*np.cos( (x_n*x)/a)) / (a * (x_n - np.sin(x_n)*np.cos(x_n)))
            eps_B += ( np.exp( (-1.0*x_n*x_n*c*t)/(a*a)) * np.sin(x_n)*np.cos(x_n)) / (x_n - np.sin(x_n)*np.cos(x_n))
            eps_C += ( np.exp( (-1.0*x_n*x_n*c*t)/(x_n*x_n)) * np.sin(x_n)*np.cos(x_n)) / (x_n - np.sin(x_n)*np.cos(x_n))

        trace_strain[t_track,:] = (F/G)*eps_A + ( (F*nu)/(2.0*G*a)) - eps_B/(G*a) - (F*(1.0-nu))/(2/0*G*a) + eps_C/(G*a)
        t_track += 1

    return trace_strain


# ==============================================================================
f = h5py.File('./output/mandel_quad-domain.h5','r')

t = f['time'][:]
t = t.ravel()

U = f['vertex_fields/displacement'][:]
P = f['vertex_fields/pressure'][:]
S = f['vertex_fields/trace_strain'][:]

pos = f['geometry/vertices'][:]

zeroArray = mandelZeros()
U_exact = displacement(pos, t, zeroArray)
P_exact = np.reshape(pressure(pos, t, zeroArray),[t.shape[0],pos.shape[0],1])
#S_exact = trace_strain(pos, t)

# Graph time snapshots
t_steps = t.ravel()
t_step_array = np.linspace(0,t_steps.size,5).astype(np.int)
t_step_array[0] += 2
t_step_array[-1] -= 1
#t_step_array = np.array([5, 10, 20, 25, 30])

t_N = c*t / (a*a)
P_N = (a*P) / F

U_N = U.copy()
U_N[:,:,0] = U[:,:,0] / a
U_N[:,:,1] = U[:,:,1] / b

P_exact_N = (a*P_exact) / F

U_exact_N = U_exact.copy()
U_exact_N[:,:,0] = U_exact[:,:,0] / a
U_exact_N[:,:,1] = U_exact[:,:,1] / b

pos_N = pos.copy()
pos_N[:,0] = pos[:,0] / a
pos_N[:,1] = pos[:,1] / b

# Zero Lines
x_zero_row = np.flatnonzero(pos_N[:,0]==0)
x_zero_pos = np.zeros([x_zero_row.size,3])
x_zero_pos[:,:2] = pos[x_zero_row]
x_zero_pos[:,2] = x_zero_row
x_zero_pos = x_zero_pos[x_zero_pos[:,1].argsort()][:,2]
x_zero_pos = x_zero_pos.astype(np.int)

z_zero_row = np.flatnonzero(pos_N[:,1]==0)
z_zero_pos = np.zeros([z_zero_row.size,3])
z_zero_pos[:,:2] = pos[z_zero_row]
z_zero_pos[:,2] = z_zero_row
z_zero_pos = z_zero_pos[z_zero_pos[:,0].argsort()][:,2]
z_zero_pos = z_zero_pos.astype(np.int)

# Initial Conditions
U_initial = U[0,:,:].copy()

U_initial[:,0] = (F*nu_u*pos[:,0])/(2.*G*a)
U_initial[:,1] = -(F*(1.-nu_u)*pos[:,1])/(2.*G*a)

U_final = U[-1,:,:].copy()

U_final[:,0] = (F*nu*pos[:,0])/(2.*G*a)
U_final[:,1] = -(F*(1.-nu)*pos[:,1])/(2.*G*a)

P_initial = P[0,:].copy()
P_initial[:] = 1./(3.*a) * B * (1. + nu_u) * F

P_final = P[-1,:].copy()
P_final[:] = 1./(3.*a) * B * (1. + nu) * F

cm_numeric = ['red','orange','green','blue','indigo', 'violet']
cm_analytic = ['red','orange','green','blue','indigo', 'violet']

# ==============================================================================
# Generate Analytical Solution Plots
# ==============================================================================

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

X = pos.copy()[:,0]
Y = pos.copy()[:,1]
X_N = pos_N.copy()[:,0]
Y_N = pos_N.copy()[:,1]



y_pos = np.where(Y == ymax)[0]
x_pos = np.where(X == xmax)[0]
x_fig = X[y_pos]
y_fig = Y[x_pos]
x_fig, tx_fig = np.meshgrid(x_fig, t)
y_fig, ty_fig = np.meshgrid(y_fig, t)

# Analytical Pressure
fig = plt.figure()
fig.set_size_inches(15,10)
ax = fig.gca(projection='3d')

surf_pressure = ax.plot_surface(x_fig, tx_fig, P_exact[:,y_pos,0], cmap=cm.coolwarm, rcount=500, ccount=500)
fig.colorbar(surf_pressure, shrink=0.5, aspect=5)

ax.set_xlabel('Distance, m')
ax.set_ylabel('Time, s')
ax.set_zlabel('Pressure, Pa')
ax.set_title("Analytical Pressure Along X Axis")

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    print_angle = "{0:0=3d}".format(angle)
    savename = 'output/mandel_analytical_pressure_x_' + print_angle + '.png'
    fig.savefig(savename,dpi = 300)    
    plt.pause(.00001)

#ax.view_init(elev=30., azim=35)
#plt.show()

# Numerical Pressure

fig = plt.figure()
fig.set_size_inches(15,10)
ax = fig.gca(projection='3d')

surf_pressure = ax.plot_surface(x_fig, tx_fig, P[:,y_pos,0], cmap=cm.coolwarm, rcount=500, ccount=500)
fig.colorbar(surf_pressure, shrink=0.5, aspect=5)

ax.set_xlabel('Distance, m')
ax.set_ylabel('Time, s')
ax.set_zlabel('Pressure, Pa')
ax.set_title("Numerical Pressure Along X Axis")


for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    print_angle = "{0:0=3d}".format(angle)
    savename = 'output/mandel_numerical_pressure_x_' + print_angle + '.png'
    fig.savefig(savename,dpi = 300)    
    plt.pause(.00001)



