#!/usr/bin/env/ python

import numpy as np
import matplotlib.pyplot as plt
import h5py

# ==============================================================================
# Computational Values
ITERATIONS = 1000
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
    z_star = z/L

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
        z_star = z/L
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
    trace_strain = np.zeros((ntpts, npts,1), dtype=np.float64)
    z = locs[:,1]
    t_track = 0

    for t in tsteps:
        z_star = z/L
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
S_exact = trace_strain(pos, t)

