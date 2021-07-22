#!/usr/bin/env/ python

import numpy as np
import matplotlib.pyplot as plt
import h5py

# ==============================================================================
# Computational Values
ITERATIONS = 2000
EPS = 1e-7

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

zmax = 10.0
zmin = 0.0
ymax = 10.0
ymin = 0.0
xmax = 10.0
xmin = 0.0
P_0 = 1.0

# Height of column, m
L = ymax - ymin
H = xmax - xmin

M = 1.0 / ( phi / K_fl + ( alpha - phi ) / K_sg ) # Pa
K_u = K_d + alpha*alpha*M # Pa,      Cheng (B.5)
#K_d = K_u - alpha*alpha*M # Pa,      Cheng (B.5)
nu = (3.0*K_d - 2.0*G) / (2.0*(3.0*K_d + G)) # -,       Cheng (B.8)
nu_u = (3.0*K_u - 2.0*G) / (2.0*(3.0*K_u + G)) # -,       Cheng (B.9)
eta = (3.0*alpha*G) /(3.0*K_d + 4.0*G) #  -,       Cheng (B.11)
S_eps = (3.0*K_u + 4.0*G) / (M*(3.0*K_d + 4.0*G)) # Pa^{-1}, Cheng (B.14)
c = (k / mu_f) / S_eps # m^2 / s, Cheng (B.16)
m_v = 1.0 / (K_d + (4./3.)*G)
p0 = ((alpha*m_v)/(alpha*alpha*m_v + S_eps)) * F
# ==============================================================================

# Series functions

def F1(z_star, t_star):
    F1 = 0.
    for m in np.arange(1,2*ITERATIONS+1,2):
        F1 += 4./(m*np.pi) * np.sin(0.5*m*np.pi*z_star)*np.exp( -(m*np.pi)**2 * t_star)
    return F1

def F2(z_star, t_star):
    F2 = 0.
    for m in np.arange(1,2*ITERATIONS+1,2):
        F2 += ( 8. / (m*np.pi)**2 )  * np.cos(0.5*m*np.pi*z_star) * (1. - np.exp( -(m * np.pi)**2 * t_star) )
    return F2

def F3(z_star, t_star):
    F3 = 0.
    for m in np.arange(1,2*ITERATIONS+1,2):
        F3 += (-4.0 / (m*np.pi*L)) * np.sin(0.5*m*np.pi*z_star) * (1.0 - np.exp( -(m*np.pi)**2 * t_star))
    return F3

def displacement(locs, tsteps):
    """
    Compute displacement field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    displacement = np.zeros((ntpts, npts, dim), dtype=np.float64)
    z = locs[:,1]
    t_track = 0
    z_star = 1 - z/L

    for t in tsteps:
        if t < 0.0:
            displacement[0,:,1] = ( (P_0*L*(1.0 - 2.0*nu_u) ) / (2.0*G*(1.0 - nu_u)) ) * (1.0 - z_star)
        else:
            t_star = (c*t) / ( (2*L)**2 )
            displacement[t_track,:,1] = ((P_0*L*(1.0 - 2.0*nu_u)) / (2.0*G*(1.0 - nu_u))) * (1.0 - z_star) + ((P_0*L*(nu_u - nu)) / (2.0*G*(1.0 - nu_u)*(1.0 - nu)))*F2(z_star, t_star)
        t_track += 1

    return displacement

def pressure(locs, tsteps):
    """
    Compute pressure field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    pressure = np.zeros((ntpts, npts), dtype=np.float64)
    z = locs[:,1]
    t_track = 0

    for t in tsteps:
        z_star = 1 - z/L
        t_star = (c*t) / (4.*L**2)
        pressure[t_track,:] = ( (P_0 * eta) / (G * S_eps) ) * F1(z_star, t_star)
        t_track += 1

    return pressure    
    
def trace_strain(locs, tsteps):
    """
    Compute trace strain field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    trace_strain = np.zeros((ntpts, npts), dtype=np.float64)
    z = locs[:,1]
    t_track = 0

    for t in tsteps:
        z_star = 1 - z/L
        t_star = (c*t) / (4*L**2)
        trace_strain[t_track,:,0] = -((P_0*L*(1.0 - 2.0*nu_u)) / (2.0*G*(1.0 - nu_u)*L)) \
                                  + ((P_0*L*(nu_u - nu)) / (2.0*G*(1.0 - nu_u)*(1.0 - nu)))*F3(z_star, t_star)
        t_track += 1

    return trace_strain   

# ==============================================================================
f = h5py.File('./output/terzaghi_quad-domain.h5','r')

t = f['time'][:]

U = f['vertex_fields/displacement'][:]
P = f['vertex_fields/pressure'][:]
S = f['vertex_fields/trace_strain'][:]

pos = f['geometry/vertices'][:]

U_exact = displacement(pos, t)
P_exact =  pressure(pos, t)
#P_exact = np.reshape(pressure(pos, t),[t.shape[0],pos.shape[0],1])
#S_exact = trace_strain(pos, t)

# Graph time snapshots
t_steps = t.ravel()
t_step_array = np.linspace(0,t_steps.size,5).astype(np.int)
t_step_array[-1] -= 1
#t_step_array = np.array([5, 10, 20, 25, 30])

t_N = c*t / (L)**2
P_N = P / p0
U_N = U / -L
P_exact_N = P_exact / p0
U_exact_N = U_exact / L
pos_N = pos / L

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

cm_numeric = ['yellow','blue','black','red','green']
cm_analytic = ['^y','^b','^k','^r','^g']


# Generate Pressure Graph, x = 0
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = t_step_array[0]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], cm_analytic[0], label='Analytical, t* = ' + np.str(t_N[tstep]) )

#t5
tstep = t_step_array[1]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], cm_analytic[1], label='Analytical, t* = ' + np.str(t_N[tstep]) )

#t10
tstep = t_step_array[2]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], cm_analytic[2], label='Analytical, t* = ' + np.str(t_N[tstep]) )

#t15
tstep = t_step_array[3]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], cm_analytic[3], label='Analytical, t* = ' + np.str(t_N[tstep]) )

#t20
tstep = t_step_array[4]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], cm_analytic[4], label='Analytical, t* = ' + np.str(t_N[tstep]) )


ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Distance, z*', ylabel='Normalized Pressure, P*', title="Terzaghi's Problem: Pressure at x = 0")
fig.tight_layout()
fig.savefig('output/terzaghi_pressure.png',dpi = 300)

fig.show()


# Generate z Displacement Graph
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = t_step_array[0]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], cm_analytic[0], label='Analytical, t* = ' + np.str(t_N[tstep]) )

#t5
tstep = t_step_array[1]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], cm_analytic[1], label='Analytical, t* = ' + np.str(t_N[tstep]) )

#t10
tstep = t_step_array[2]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos, 1], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos, 1], cm_analytic[2], label='Analytical, t* = ' + np.str(t_N[tstep]) )

#t15
tstep = t_step_array[3]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos, 1], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos, 1], cm_analytic[3], label='Analytical, t* = ' + np.str(t_N[tstep]) )

#t20
tstep = t_step_array[4]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep]) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], cm_analytic[4], label='Analytical, t* = ' + np.str(t_N[tstep]) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Distance, z*', ylabel='Normalized Displacement, U*', title="Terzaghi's Problem: Z Displacement at x = 0")
fig.tight_layout()
fig.savefig('output/terzaghi_displacement.png',dpi = 300)

fig.show()


