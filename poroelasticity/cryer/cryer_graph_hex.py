#!/usr/bin/env/ python

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import h5py

np.set_printoptions(suppress=True)
# ==============================================================================
# Computational Values
ITERATIONS = 50
EPS = 1e-20

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
P_0 = 1.0
R_0 = 1.0
ndim = 3

M    = 1.0 / ( phi / K_fl + (alpha - phi) /K_sg)
kappa = k/mu_f
K_u = K_d + alpha*alpha*M
S = (3*K_u + 4*G) / (M*(3*K_d + 4*G)) #(1/M) + ( (3*alpha*alpha) / (3*K_d + 4*G) )#
c = kappa / S
nu = (3*K_d - 2*G) / (2*(3*K_d + G))
nu_u = (3*K_u - 2*G) / (2*(3*K_u + G))
U_R_inf = -1.*(P_0*R_0*(1.-2.*nu))/(2.*G*(1.+nu))
U_R_zero = -1.*(P_0*R_0*(1.-2.*nu_u))/(2.*G*(1.+nu_u))
eta = alpha*(1-2*nu) / (2*(1-nu))

# For Center
grav = 9.80665
S_c = phi*(1/K_fl) + (alpha - phi)*(1/K_sg)
eta_c = ( (K_d + 4/3*G)/(2*G) ) * (1 + (K_d*S)/(alpha*alpha))
gamma_f_c = S_c*phi*rho_f*grav + (1-phi)*rho_s*grav
c_v_c = (k / gamma_f_c) * ( (K_d + 4/3*G)/(alpha*alpha + S*(K_d + 4/3*G) ) )
# ==============================================================================
# Generate positive solutions to characteristic equation

def cryer_zeros_python(nu, nu_u, n_series=50):

    f      = lambda x: np.tan(np.sqrt(x)) - (6*(nu_u - nu)*np.sqrt(x))/(6*(nu_u - nu) - (1 - nu)*(1 + nu_u)*x) # Compressible Constituents 

    a_n = np.zeros(n_series) # initializing roots array
    xtol = 1e-30
    rtol = 1e-15
    for i in range(1,n_series+1):
        a = np.square(i*np.pi) - (i+1)*np.pi
        b = np.square(i*np.pi) + (i+1)*np.pi
        # print('a: ',a)
        # print('b: ',b) 
        f_c = 10
        f_c_old = 0
        rtol_flag = False
        c = 0
        it = 0
        while np.abs(f_c) > xtol and rtol_flag == False:
            c = (a + b) / 2
            f_c = f(c)

            # print('c: ',c)
            # print('f(c):',f_c)
                        
            if f(a)*f_c < 0:
                a = a
                b = c
            elif f(b)*f_c < 0:
                a = c
                b = b     
            else:
                print('Bisection method failed')
                # print('a: ',a)
                # print('b: ',b)
                break
            if np.abs(np.abs(f_c_old) - np.abs(f_c)) < rtol:
                rtol_flag = True                    
            it += 1
            # print('f(c): ',f_c)
            # print('rtol: ',np.abs(np.abs(f_c_old) - np.abs(f_c)))
            # print('rtol flag: ',rtol_flag)
            f_c_old = f_c
        # print('n: ',i)                
        # print('c: ',c)
        # print('f(c):',f_c)
        # print('iter: ',it)
        a_n[i-1] = c               
        
    return(a_n)    
                
def cryer_zeros(nu, nu_u,n_series=50):
    """
    This is somehow tricky, we have to solve the equation numerically in order to
    find all the positive solutions to the equation. Later we will use them to 
    compute the infinite sums. Experience has shown that 200 roots are more than enough to
    achieve accurate results. Note that we find the roots using the bisection method.
    """
    f      = lambda x: np.tan(np.sqrt(x)) - (6*(nu_u - nu)*np.sqrt(x))/(6*(nu_u - nu) - (1 - nu)*(1 + nu_u)*x) # Compressible Constituents 
#    f      = lambda x: np.tan(np.sqrt(x)) - (2*(1-2*nu)*np.sqrt(x))/(2*(1-2*nu) - (1-nu)*x) # Incompressible Constituents

    a_n = np.zeros(n_series) # initializing roots array
    for i in range(1,len(a_n)+1):
        a1 = np.square(i*np.pi) - (i+1)*np.pi
        a2 = np.square(i*np.pi) + (i+1)*np.pi        
        a_n[i-1] = opt.bisect( f,                           # function
                               a1,                          # left point 
                               a2,                          # right point (a tiny bit less than pi/2)
                               xtol=1e-30,                  # absolute tolerance
                               rtol=1e-15                   # relative tolerance
                           )  
    
    return a_n

def pressure(locs, tsteps, x_n):
    """
    Compute pressure field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    pressure = np.zeros((ntpts, npts), dtype=np.float64)
    
    center = np.where(~locs.any(axis=1))[0]
    R = np.sqrt(locs[:,0]*locs[:,0] + locs[:,1]*locs[:,1] + locs[:,2]*locs[:,2])
    R_star = R.reshape([R.size,1]) / R_0
    x_n.reshape([1,x_n.size])

    E = np.square(1-nu)*np.square(1+nu_u)*x_n - 18*(1+nu)*(nu_u-nu)*(1-nu_u)
    
    t_track = 0

    for t in tsteps:
        t_star = (c*t)/(R_0**2)
       
        pressure[t_track,:] = np.sum( ( (18*np.square(nu_u-nu) ) / (eta*E) ) * \
                                    ( (np.sin(R_star*np.sqrt(x_n))) / (R_star*np.sin(np.sqrt(x_n)) ) - 1 ) * \
                                    np.exp(-x_n*t_star) , axis=1) 

        # Account for center value
        #pressure[t_track,center] = np.sum( (8*eta*(np.sqrt(x_n) - np.sin(np.sqrt(x_n)))) / ( (x_n - 12*eta + 16*eta*eta)*np.sin(np.sqrt(x_n)) ) * np.exp(-x_n * t_star) )
        pressure[t_track,center] = np.sum( ( (18*np.square(nu_u-nu) ) / (eta*E) ) * \
                                    ( (np.sqrt(x_n)) / (np.sin(np.sqrt(x_n)) ) - 1 ) * \
                                    np.exp(-x_n*t_star)) 
        t_track += 1

    return pressure
    
def displacement(locs, tsteps, x_n):
    """
    Compute displacement field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    disp = np.zeros((ntpts, npts), dtype=np.float64)    
    
    center = np.where(~locs.any(axis=1))[0]
    R = np.sqrt(locs[:,0]*locs[:,0] + locs[:,1]*locs[:,1] + locs[:,2]*locs[:,2])
    R_star = R.reshape([R.size,1]) / R_0
    x_n.reshape([1,x_n.size])
    
    E = np.square(1-nu)*np.square(1+nu_u)*x_n - 18*(1+nu)*(nu_u-nu)*(1-nu_u)
    
    t_track = 0
    
    for t in tsteps:
        t_star = (c*t)/(R_0**2)
        disp[t_track, :] = R_star.ravel() - np.nan_to_num(np.sum(((12*(1 + nu)*(nu_u - nu)) / \
                                           ((1 - 2*nu)*E*R_star*R_star*x_n*np.sin(np.sqrt(x_n))) ) * \
                                    (3*(nu_u - nu) * (np.sin(R_star*np.sqrt(x_n)) - R_star*np.sqrt(x_n)*np.cos(R_star*np.sqrt(x_n))) + \
                                    (1 - nu)*(1 - 2*nu)*R_star*R_star*R_star*x_n*np.sin(np.sqrt(x_n))) * \
                                    np.exp(-x_n*t_star),axis=1))
                                        
        t_track += 1
    
    return disp

def displacement_small_time(locs, tsteps, x_n):
    """
    Compute displacement field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    disp = np.zeros((ntpts, npts), dtype=np.float64)    
    
    center = np.where(~locs.any(axis=1))[0]
    R = np.sqrt(locs[:,0]*locs[:,0] + locs[:,1]*locs[:,1] + locs[:,2]*locs[:,2])
    R_star = R.reshape([R.size,1]) / R_0
    x_n.reshape([1,x_n.size])
    
    E = np.square(1-nu)*np.square(1+nu_u)*x_n - 18*(1+nu)*(nu_u-nu)*(1-nu_u)
    
    t_track = 0
    
    for t in tsteps:
        t_star = (c*t)/(R_0**2)
        disp[t_track, :] = R_star.ravel() + np.nan_to_num( ( 12*(nu_u-nu) ) / ( np.sqrt(np.pi)*(1-nu)*(1+nu_u) )  * \
                                            R_star.ravel()*np.sqrt(t_star) )
                                       
        t_track += 1
    
    return disp

def E_func(x_n, nu, nu_u):
    E = (1-nu)*(1-nu)*(1+nu_u)*(1+nu_u)*x_n - 18*(1+nu)*(nu_u-nu)*(1-nu_u)
    return E
# ==============================================================================

# ==============================================================================
f = h5py.File('./output/cryer_hex-poroelastic.h5','r')

t = f['time'][:]
t = t.ravel()

U = f['vertex_fields/displacement'][:]
P = f['vertex_fields/pressure'][:]
E = f['vertex_fields/trace_strain'][:]

pos = f['geometry/vertices'][:]
R = np.sqrt(pos[:,0]*pos[:,0] + pos[:,1]*pos[:,1] + pos[:,2]*pos[:,2])
theta_sph = np.nan_to_num( np.arctan( np.nan_to_num( np.sqrt(pos[:,0]**2 + pos[:,1]**2) / pos[:,2] ) ) )
phi_sph = np.nan_to_num( np.arctan( np.nan_to_num( pos[:,1] / pos[:,0] ) ) )

# Transform position to spherical coordinates
pos_sph = np.zeros(pos.shape)
#U_sph = np.zeros(U.shape)
# (r, theta, phi)
pos_sph[:,0] = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
pos_sph[:,1] = np.nan_to_num( np.arctan( np.nan_to_num( np.sqrt(pos[:,0]**2 + pos[:,1]**2) / pos[:,2] ) ) )
pos_sph[:,2] = np.nan_to_num( np.arctan( np.nan_to_num( pos[:,1] / pos[:,0] ) ) )

#pos_sph = np.nan_to_num(pos_sph, nan=0.0)
#U_sph = np.nan_to_num(U_sph, nan=0.0)

t_N = (c*t) / R_0**2
P_N = P / P_0

U_R = -np.sqrt(U[:,:,0]*U[:,:,0] + U[:,:,1]*U[:,:,1] + U[:,:,2]*U[:,:,2])

U_R_inf = -( (P_0*R_0*(1-2*nu))/(2*G*(1+nu)))
U_R_N = U_R/U_R_inf

zeroArray = cryer_zeros_python(nu,nu_u,ITERATIONS)
P_exact_N = np.reshape(pressure(pos, t, zeroArray),[t.shape[0],pos.shape[0],1])
U_exact_R_N = np.reshape(displacement(pos, t, zeroArray),[t.shape[0],pos.shape[0],1])
U_exact_R = U_exact_R_N * U_R_inf
U_exact_R_N_ST = np.reshape(displacement_small_time(pos, t, zeroArray),[t.shape[0],pos.shape[0],1])

U_exact_R_ST = U_exact_R_N * U_R_zero
U_exact_N_ST = (U_exact_R_N_ST * U_R_zero) / U_R_inf

# Cartesian analytical data
U_exact = np.zeros(U.shape)
U_exact[:,:,0] = U_exact_R[:,:,0] * np.cos(pos_sph[:,2]) * np.sin(pos_sph[:,1])
U_exact[:,:,1] = U_exact_R[:,:,0] * np.sin(pos_sph[:,2]) * np.sin(pos_sph[:,1])
U_exact[:,:,2] = U_exact_R[:,:,0] * np.cos(pos_sph[:,1])


# Select linear samples

#x_slice = np.where(~pos[:,1:].any(axis=1))[0]

x_slice = np.nonzero(np.logical_and(pos[:,1] == 0.0, pos[:,2] == 0.0))[0]
x_arr1 = R[x_slice]
x_arr1inds = x_arr1.argsort()
x_slice = x_slice[x_arr1inds[::1]]

y_slice = np.nonzero(np.logical_and(pos[:,0] == 0.0, pos[:,2] == 0.0))[0]
y_arr1 = R[y_slice]
y_arr1inds = y_arr1.argsort()
y_slice = y_slice[y_arr1inds[::1]]

#z_slice = np.where(~pos[:,:2].any(axis=1))[0]

z_slice = np.nonzero(np.logical_and(pos[:,0] == 0.0, pos[:,1] == 0.0))[0]
z_arr1 = R[z_slice]
z_arr1inds = z_arr1.argsort()
z_slice = z_slice[z_arr1inds[::1]]

center = np.where(~pos.any(axis=1))[0]


# Graph time snapshots
t_steps = t.ravel()
n_graph_steps = 10
t_step_array = np.linspace(0,t_steps.size,n_graph_steps).astype(np.int)
t_step_array[0] += 2
t_step_array[-1] -= 1
n_steps = t_N.size


cm_numeric = ['red','orange','green','blue','indigo', 'violet']
cm_analytic = ['red','orange','green','blue','indigo', 'violet']


# Pore Pressure at Center
fig, ax = plt.subplots()
fig.set_size_inches(15,10)
ax.semilogx(t_N, P_N[:,center,0], color=cm_numeric[1], label='Numerical')
ax.semilogx(t_N, P_exact_N[:,center,0], cm_analytic[1],marker='^', linestyle=' ', label='Analytical')

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Time, t*', ylabel='Normalized Pressure, P*', title="Cryer's Problem: Normalized Pressure at Center, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_pressure_at_center_hex.png',dpi = 300)
fig.show()

# ==============================================================================
# Normalized Pore Pressure along x axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(R[x_slice], P_N[tstep, x_slice,0], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], P_exact_N[tstep, x_slice,0], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t2
tstep = 6 # np.int(n_steps/5 * 1)
ax.plot(R[x_slice], P_N[tstep, x_slice,0], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], P_exact_N[tstep, x_slice,0], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t5
tstep = 14 # np.int(n_steps/5 * 2)
ax.plot(R[x_slice], P_N[tstep, x_slice,0], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], P_exact_N[tstep, x_slice,0], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t25
tstep = 20 # np.int(n_steps/5 * 3)
ax.plot(R[x_slice], P_N[tstep, x_slice,0], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], P_exact_N[tstep, x_slice,0], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t50
tstep = 35 # np.int(n_steps/5 * 4)
ax.plot(R[x_slice], P_N[tstep, x_slice,0], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], P_exact_N[tstep, x_slice,0], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t70
tstep = -1
ax.plot(R[x_slice], P_N[tstep, x_slice,0], color=cm_numeric[5], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], P_exact_N[tstep, x_slice,0], color=cm_analytic[5], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )


ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Radial Distance, R*', ylabel='Normalized Pressure, P*', title="Cryer's Problem: Normalized Pressure Along X Axis, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_pressure_along_x_axis_norm_hex.png',dpi = 300)
fig.show()

# ==============================================================================
# Normalized Radial Displacement along x axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(R[x_slice], U_R_N[tstep, x_slice], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N[tstep, x_slice,0], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N_ST[tstep, x_slice,0], color=cm_analytic[0], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t2
tstep = 6 # np.int(n_steps/5 * 1)
ax.plot(R[x_slice], U_R_N[tstep, x_slice], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N[tstep, x_slice,0], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N_ST[tstep, x_slice,0], color=cm_analytic[1], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t5
tstep = 14 # np.int(n_steps/5 * 2)
ax.plot(R[x_slice], U_R_N[tstep, x_slice], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N[tstep, x_slice,0], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N_ST[tstep, x_slice,0], color=cm_analytic[2], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t25
tstep = 20 # np.int(n_steps/5 * 3)
ax.plot(R[x_slice], U_R_N[tstep, x_slice], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N[tstep, x_slice,0], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N_ST[tstep, x_slice,0], color=cm_analytic[3], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t50
tstep = 35 # np.int(n_steps/5 * 4)
ax.plot(R[x_slice], U_R_N[tstep, x_slice], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N[tstep, x_slice,0], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N_ST[tstep, x_slice,0], color=cm_analytic[4], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t70
tstep = -1
ax.plot(R[x_slice], U_R_N[tstep, x_slice], color=cm_numeric[5], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N[tstep, x_slice,0], color=cm_analytic[5], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_N_ST[tstep, x_slice,0], color=cm_analytic[5], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Radial Distance, R*', ylabel='Normalized Radial Displacement, U*', title="Cryer's Problem: Normalized Radial Displacement Along X Axis, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_radial_displacement_along_x_axis_norm_hex.png',dpi = 300)
fig.show()

# ==============================================================================
# Radial Displacement along x axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(R[x_slice], U_R[tstep, x_slice], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R[tstep, x_slice,0], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_ST[tstep, x_slice,0], color=cm_analytic[0], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t2
tstep = 6 # np.int(n_steps/5 * 1)
ax.plot(R[x_slice], U_R[tstep, x_slice], color=cm_numeric[1], label='Numerical, t = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R[tstep, x_slice,0], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_ST[tstep, x_slice,0], color=cm_analytic[1], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = 14 # np.int(n_steps/5 * 2)
ax.plot(R[x_slice], U_R[tstep, x_slice], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R[tstep, x_slice,0], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_ST[tstep, x_slice,0], color=cm_analytic[2], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t25
tstep = 20 # np.int(n_steps/5 * 3)
ax.plot(R[x_slice], U_R[tstep, x_slice], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R[tstep, x_slice,0], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_ST[tstep, x_slice,0], color=cm_analytic[3], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t50
tstep = 35 # np.int(n_steps/5 * 4)
ax.plot(R[x_slice], U_R[tstep, x_slice], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R[tstep, x_slice,0], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_ST[tstep, x_slice,0], color=cm_analytic[4], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t70
tstep = -1
ax.plot(R[x_slice], U_R[tstep, x_slice], color=cm_numeric[5], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R[tstep, x_slice,0], color=cm_analytic[5], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact_R_ST[tstep, x_slice,0], color=cm_analytic[5], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Radial Distance, R', ylabel='Radial Displacement, U', title="Cryer's Problem: Radial Displacement Along X Axis, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_radial_displacement_along_x_axis_hex.png',dpi = 300)
fig.show()

# ==============================================================================
# Normalized Radial Displacement along z axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(R[z_slice], U_R_N[tstep, z_slice], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N[tstep, z_slice,0], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N_ST[tstep, z_slice,0], color=cm_analytic[0], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t2
tstep = 6 # np.int(n_steps/5 * 1)
ax.plot(R[z_slice], U_R_N[tstep, z_slice], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N[tstep, z_slice,0], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N_ST[tstep, z_slice,0], color=cm_analytic[1], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t5
tstep = 14 # np.int(n_steps/5 * 2)
ax.plot(R[z_slice], U_R_N[tstep, z_slice], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N[tstep, z_slice,0], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N_ST[tstep, z_slice,0], color=cm_analytic[2], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t25
tstep = 20 # np.int(n_steps/5 * 3)
ax.plot(R[z_slice], U_R_N[tstep, z_slice], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N[tstep, z_slice,0], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N_ST[tstep, z_slice,0], color=cm_analytic[3], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t50
tstep = 35 # np.int(n_steps/5 * 4)
ax.plot(R[z_slice], U_R_N[tstep, z_slice], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N[tstep, z_slice,0], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N_ST[tstep, z_slice,0], color=cm_analytic[4], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

#t70
tstep = -1
ax.plot(R[z_slice], U_R_N[tstep, z_slice], color=cm_numeric[5], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N[tstep, z_slice,0], color=cm_analytic[5], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_N_ST[tstep, z_slice,0], color=cm_analytic[5], marker='+', linestyle=' ',  label='Analytical ST, t* = ' + np.str(t_N[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Radial Distance, R*', ylabel='Normalized Radial Displacement, U*', title="Cryer's Problem: Normalized Radial Displacement Along Z Axis, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_radial_displacement_along_z_axis_norm_hex.png',dpi = 300)
fig.show()

# ==============================================================================
# Radial Displacement along z axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(R[z_slice], U_R[tstep, z_slice], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R[tstep, z_slice,0], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_ST[tstep, z_slice,0], color=cm_analytic[0], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t2
tstep = 6 # np.int(n_steps/5 * 1)
ax.plot(R[z_slice], U_R[tstep, z_slice], color=cm_numeric[1], label='Numerical, t = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R[tstep, z_slice,0], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_ST[tstep, z_slice,0], color=cm_analytic[1], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = 14 # np.int(n_steps/5 * 2)
ax.plot(R[z_slice], U_R[tstep, z_slice], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R[tstep, z_slice,0], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_ST[tstep, z_slice,0], color=cm_analytic[2], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t25
tstep = 20 # np.int(n_steps/5 * 3)
ax.plot(R[z_slice], U_R[tstep, z_slice], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R[tstep, z_slice,0], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_ST[tstep, z_slice,0], color=cm_analytic[3], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t50
tstep = 35 # np.int(n_steps/5 * 4)
ax.plot(R[z_slice], U_R[tstep, z_slice], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R[tstep, z_slice,0], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_ST[tstep, z_slice,0], color=cm_analytic[5], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

#t70
tstep = -1
ax.plot(R[z_slice], U_R[tstep, z_slice], color=cm_numeric[5], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R[tstep, z_slice,0], color=cm_analytic[5], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact_R_ST[tstep, z_slice,0], color=cm_analytic[5], marker='+', linestyle=' ',  label='Analytical ST, t = ' + np.str(t[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Radial Distance, R', ylabel='Radial Displacement, U', title="Cryer's Problem: Radial Displacement Along Z Axis, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_radial_displacement_along_z_axis_hex.png',dpi = 300)
fig.show()

# ==============================================================================
# X Displacement along x axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(R[x_slice], U[tstep, x_slice, 0], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact[tstep, x_slice, 0], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t2
tstep = 6 # np.int(n_steps/5 * 1)
ax.plot(R[x_slice], U[tstep, x_slice, 0], color=cm_numeric[1], label='Numerical, t = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[x_slice], U_exact[tstep, x_slice, 0], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = 14 # np.int(n_steps/5 * 2)
ax.plot(R[x_slice], U[tstep, x_slice, 0], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact[tstep, x_slice, 0], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t25
tstep = 20 # np.int(n_steps/5 * 3)
ax.plot(R[x_slice], U[tstep, x_slice, 0], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact[tstep, x_slice, 0], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t50
tstep = 35 # np.int(n_steps/5 * 4)
ax.plot(R[x_slice], U[tstep, x_slice, 0], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact[tstep, x_slice, 0], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t70
tstep = -1
ax.plot(R[x_slice], U[tstep, x_slice, 0], color=cm_numeric[5], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[x_slice], U_exact[tstep, x_slice, 0], color=cm_analytic[5], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Distance, X', ylabel='Displacement, U', title="Cryer's Problem: X Displacement Along X Axis, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_displacement_along_x_axis_hex.png',dpi = 300)
fig.show()

# ==============================================================================
# Y Displacement along y axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(R[y_slice], U[tstep, y_slice, 1], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[y_slice], U_exact[tstep, y_slice, 1], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t2
tstep = 6 # np.int(n_steps/5 * 1)
ax.plot(R[y_slice], U[tstep, y_slice, 1], color=cm_numeric[1], label='Numerical, t = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[y_slice], U_exact[tstep, y_slice, 1], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = 14 # np.int(n_steps/5 * 2)
ax.plot(R[y_slice], U[tstep, y_slice, 1], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[y_slice], U_exact[tstep, y_slice, 1], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t25
tstep = 20 # np.int(n_steps/5 * 3)
ax.plot(R[y_slice], U[tstep, y_slice, 1], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[y_slice], U_exact[tstep, y_slice, 1], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t50
tstep = 35 # np.int(n_steps/5 * 4)
ax.plot(R[y_slice], U[tstep, y_slice, 1], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[y_slice], U_exact[tstep, y_slice, 1], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t70
tstep = -1
ax.plot(R[y_slice], U[tstep, y_slice, 1], color=cm_numeric[5], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[y_slice], U_exact[tstep, y_slice, 1], color=cm_analytic[5], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Distance, Y', ylabel='Displacement, U', title="Cryer's Problem: Y Displacement Along Y Axis, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_displacement_along_y_axis_hex.png',dpi = 300)
fig.show()

# ==============================================================================
# Z Displacement along z axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(R[z_slice], U[tstep, z_slice, 2], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact[tstep, z_slice, 2], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t2
tstep = 6 # np.int(n_steps/5 * 1)
ax.plot(R[z_slice], U[tstep, z_slice, 2], color=cm_numeric[1], label='Numerical, t = ' + np.str(t_N[tstep].ravel()) )
ax.plot(R[z_slice], U_exact[tstep, z_slice, 2], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = 14 # np.int(n_steps/5 * 2)
ax.plot(R[z_slice], U[tstep, z_slice, 2], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact[tstep, z_slice, 2], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t25
tstep = 20 # np.int(n_steps/5 * 3)
ax.plot(R[z_slice], U[tstep, z_slice, 2], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact[tstep, z_slice, 2], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t50
tstep = 35 # np.int(n_steps/5 * 4)
ax.plot(R[z_slice], U[tstep, z_slice, 2], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact[tstep, z_slice, 2], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t70
tstep = -1
ax.plot(R[z_slice], U[tstep, z_slice, 2], color=cm_numeric[5], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(R[z_slice], U_exact[tstep, z_slice, 2], color=cm_analytic[5], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Distance, Z', ylabel='Displacement, U', title="Cryer's Problem: Z Displacement Along Z Axis, Hex Mesh")
fig.tight_layout()
fig.savefig('output/cryer_displacement_along_z_axis_hex.png',dpi = 300)
fig.show()



