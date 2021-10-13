#!/usr/bin/env/ python

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import h5py

# ==============================================================================
# Computational Values
ITERATIONS = 300
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

ndim = 2

zmax = 1.0
zmin = 0.0
ymax = 1.0
ymin = 0.0
xmax = 10.0
xmin = 0.0



M    = 1.0 / ( phi / K_fl + (alpha - phi) /K_sg)
K_u = K_d + alpha*alpha*M
nu = (3*K_d - 2*G) / (2*(3*K_d + G))
nu_u = (3*K_u - 2*G) / (2*(3*K_u + G))
kappa = k / mu_f
a = (xmax - xmin)
b = (ymax - ymin)

F = 1.0*a
P_0 = F

B_alt = (3 * (nu_u - nu) )/(alpha*(1-2*nu)*(1+nu_u))
B = (alpha*M) / (K_d + alpha*alpha*M) 
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

K_nu = 2*G*( (1-nu) / (1-2*nu) )
c = kappa*M*( K_nu / (K_nu + alpha*alpha*M) )
# ==============================================================================
def mandelZeros():
    """
    Compute roots for analytical mandel problem solutions
    """
    zeroArray = np.zeros(ITERATIONS)
    x0 = 0
    
    for i in np.arange(0, ITERATIONS):
        a1 = x0+np.pi/4
        a2 = x0+np.pi/2 - 10000*2.2204e-16

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
        x0 += np.pi
    return zeroArray

def displacement(locs,tsteps,a_n):
    """
    Compute displacement field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    displacement = np.zeros((ntpts, npts, dim), dtype=np.float64)
    x = locs[:,0]
    z = locs[:,1]
    t_track = 0

    for t in tsteps:

        ux_sum_A = np.sum( ( np.sin(a_n)*np.cos(a_n) / ( a_n - np.sin(a_n)*np.cos(a_n) ) ) * \
                       np.exp( -(a_n*a_n*c*t)/(a*a) ) )

        ux_sum_B = np.sum( ( np.cos(a_n) / ( a_n - np.sin(a_n)*np.cos(a_n) ) ) * \
                       np.sin( (a_n*x.reshape([x.size,1]) ) / a ) * \
                       np.exp(-(a_n*a_n*c*t)/(a*a) ),axis=1)
            
        uy_sum   = np.sum( (np.sin(a_n)*np.cos(a_n) / (a_n - np.sin(a_n)*np.cos(a_n))) * \
                       np.exp( -(a_n*a_n*c*t)/(a*a) ) )


        # Isotropic Formulation
        displacement[t_track,:,0] = (  ( F * nu ) / ( 2 * G * a ) - ( F * nu_u ) / ( G * a ) * ux_sum_A ) * x + ( F / G ) * ux_sum_B
        displacement[t_track,:,1] = ( -( F * ( 1 - nu ) ) / ( 2 * G * a ) + ( F * ( 1 - nu_u ) ) / ( G * a ) * uy_sum ) * z

        t_track += 1

    return displacement

def displacement_AB(locs,tsteps,zeroArray):
    """
    Compute displacement field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    displacement = np.zeros((ntpts, npts, dim), dtype=np.float64)
    x = locs[:,0]
    z = locs[:,1]
    t_track = 0
    A_array = np.zeros(ntpts)
    B_array = np.zeros([x.size,ntpts])
    for t in tsteps:

        A_array[t_track] = np.sum( (np.sin(zeroArray)*np.cos(zeroArray) / (zeroArray - np.sin(zeroArray)*np.cos(zeroArray))) * \
                       np.exp( -(zeroArray*zeroArray*c*t)/(a*a) ) )

        B_array[:,t_track] = np.sum( (np.cos(zeroArray) / (zeroArray - np.sin(zeroArray)*np.cos(zeroArray))) * \
                       np.sin( (zeroArray*x.reshape([x.size,1]))/a) * \
                       np.exp(-(zeroArray*zeroArray*c*t)/(a*a)),axis=1)

        t_track += 1

    return A_array, B_array



def pressure(locs, tsteps, a_n):
    """
    Compute pressure field at locations.
    """
    (npts, dim) = locs.shape
    ntpts = tsteps.shape[0]
    pressure = np.zeros((ntpts, npts), dtype=np.float64)
    x = locs[:,0]
    z = locs[:,1]
    t_track = 0

    for t in tsteps:
        p = np.sum( np.sin(a_n) / ( a_n - np.sin(a_n) * np.cos(a_n) ) * \
                  ( np.cos( ( a_n * x.reshape( [ x.size, 1 ] ) ) / a ) - np.cos(a_n) ) * \
                  np.exp( -(a_n*a_n*c*t)/(a*a)), axis=1 )
        
        pressure[t_track,:] = ( 2 * F * B * ( 1 + nu_u ) ) / ( 3 * a ) * p
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
# Generate positive solutions to characteristic equation

def mandel_zeros(nu, nu_u,n_series=200):
    # Solutions to tan(x) - ((1-nu)/(nu_u-nu)) x = 0
    """
    This is somehow tricky, we have to solve the equation numerically in order to
    find all the positive solutions to the equation. Later we will use them to 
    compute the infinite sums. Experience has shown that 200 roots are more than enough to
    achieve accurate results. Note that we find the roots using the bisection method.
    """
    f      = lambda x: np.tan(x) - ((1-nu)/(nu_u-nu))*x # define the algebraic eq. as a lambda function
#    n_series = 200           # number of roots
    a_n = np.zeros(n_series) # initializing roots array
    x0 = 0                   # initial point
    for i in range(0,len(a_n)):
        a_n[i] = opt.bisect( f,                           # function
                             x0+np.pi/4,                  # left point 
                             x0+np.pi/2-10000*2.2204e-16, # right point (a tiny bit less than pi/2)
                             xtol=1e-30,                  # absolute tolerance
                             rtol=1e-15                   # relative tolerance
                           )        
        x0 += np.pi # apply a phase change of pi to get the next root
    
    return a_n
    
def mandel_zeros_python(nu, nu_u, n_series=50):

    f      = lambda x: np.tan(x) - ((1-nu)/(nu_u-nu))*x # define the algebraic eq. as a lambda function

    a_n = np.zeros(n_series) # initializing roots array
    xtol = 1e-30
    rtol = 1e-15
    for i in range(1,n_series+1):
        a = (i-1)*np.pi + np.pi/4
        b = (i-1)*np.pi + np.pi/2 - 10000*2.2204e-16
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
# ==============================================================================

# ==============================================================================
f = h5py.File('./output/step01_quad-poroelastic.h5','r')

t = f['time'][:]
t = t.ravel()

U = f['vertex_fields/displacement'][:]
P = f['vertex_fields/pressure'][:]
S = f['vertex_fields/trace_strain'][:]

pos = f['geometry/vertices'][:]

#zeroArray = mandelZeros()
zeroArray = mandel_zeros_python(nu,nu_u,ITERATIONS)
U_exact = displacement(pos, t, zeroArray)
P_exact = np.reshape(pressure(pos, t, zeroArray),[t.shape[0],pos.shape[0],1])
#S_exact = trace_strain(pos, t)

# Graph time snapshots
#t_steps = t.ravel()
#t_step_array = np.linspace(0,t_steps.size,5).astype(np.int)
#t_step_array[0] += 2
#t_step_array[-1] -= 1

# Define snapshots as per Cheng and Detournay paper
t_N_step_array = np.array([0.01, 0.1, 0.5, 1.0, 2.0])
t_step_array_exact = (t_N_step_array*a*a) / c

t_step_array = np.zeros(t_step_array_exact.size)

for item in np.arange(0,t_step_array_exact.size):
    t_step_array[item] = np.abs(t - t_step_array_exact[item]).argmin()

t_step_array[0] = 0
t_step_array = t_step_array.astype(np.int)    



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
P_initial[:] = 1/(3*a) * B * (1 + nu_u) * F

P_final = P[-1,:].copy()
P_final[:] = 1./(3.*a) * B * (1. + nu) * F

cm_numeric = ['red','orange','green','blue','indigo', 'violet']
cm_analytic = ['red','orange','green','blue','indigo', 'violet']

# Y Displacement at ypos
ypos_loc = np.flatnonzero(pos[:,1]==ymax)
ypos_y = U_exact[:,np.flatnonzero(pos[:,1]==ymax),1]


# ==============================================================================
# Raw Output
# ==============================================================================

# Generate X Displacement Graph
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(pos[z_zero_pos,0], U[tstep, z_zero_pos,0], color=cm_numeric[-1], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], U_exact[tstep, z_zero_pos,0], color=cm_analytic[-1], marker='^', linestyle=' ', label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t1
tstep = t_step_array[0]
ax.plot(pos[z_zero_pos,0], U[tstep, z_zero_pos,0], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], U_exact[tstep, z_zero_pos,0], color=cm_analytic[0], marker='^', linestyle=' ', label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = t_step_array[1]
ax.plot(pos[z_zero_pos,0], U[tstep, z_zero_pos,0], color=cm_numeric[1], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], U_exact[tstep, z_zero_pos,0], color=cm_analytic[1], marker='^', linestyle=' ', label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t10
tstep = t_step_array[2]
ax.plot(pos[z_zero_pos,0], U[tstep, z_zero_pos,0], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], U_exact[tstep, z_zero_pos,0], color=cm_analytic[2], marker='^', linestyle=' ', label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t15
tstep = t_step_array[3]
ax.plot(pos[z_zero_pos,0], U[tstep, z_zero_pos,0], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], U_exact[tstep, z_zero_pos,0], color=cm_analytic[3], marker='^', linestyle=' ', label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t20
tstep = t_step_array[4]
ax.plot(pos[z_zero_pos,0], U[tstep, z_zero_pos,0], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], U_exact[tstep, z_zero_pos,0], color=cm_analytic[4], marker='^', linestyle=' ', label='Analytical, t = ' + np.str(t[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Distance, m', ylabel='Displacement, m', title="Mandel's Problem: X Displacement along z = 0")
fig.tight_layout()
fig.savefig('output/mandel_displacement_x_raw.png',dpi = 300)

fig.show()

# Generate Pressure Graph, Z = 0
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(pos[z_zero_pos,0], P[tstep, z_zero_pos], color=cm_numeric[-1], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], P_exact[tstep, z_zero_pos], color=cm_analytic[-1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t1
tstep = t_step_array[0]
ax.plot(pos[z_zero_pos,0], P[tstep, z_zero_pos], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], P_exact[tstep, z_zero_pos], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = t_step_array[1]
ax.plot(pos[z_zero_pos,0], P[tstep, z_zero_pos], color=cm_numeric[1], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], P_exact[tstep, z_zero_pos], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t10
tstep = t_step_array[2]
ax.plot(pos[z_zero_pos,0], P[tstep, z_zero_pos], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], P_exact[tstep, z_zero_pos], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t15
tstep = t_step_array[3]
ax.plot(pos[z_zero_pos,0], P[tstep, z_zero_pos], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], P_exact[tstep, z_zero_pos], cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t20
tstep = t_step_array[4]
ax.plot(pos[z_zero_pos,0], P[tstep, z_zero_pos], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[z_zero_pos,0], P_exact[tstep, z_zero_pos], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Distance, m', ylabel='Pressure, Pa', title="Mandel's Problem: Pressure along z = 0")
fig.tight_layout()
fig.savefig('output/mandel_pressure_z_0_raw.png',dpi = 300)

fig.show()

# Generate Pressure Graph, X = 0
fig, ax = plt.subplots()
fig.set_size_inches(15,10)


#t0
tstep = 0
ax.plot(pos[x_zero_pos,1], P[tstep, x_zero_pos], color=cm_numeric[-1], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], P_exact[tstep, x_zero_pos], color=cm_analytic[-1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t1
tstep = t_step_array[0]
ax.plot(pos[x_zero_pos,1], P[tstep, x_zero_pos], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], P_exact[tstep, x_zero_pos], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = t_step_array[1]
ax.plot(pos[x_zero_pos,1], P[tstep, x_zero_pos], color=cm_numeric[1], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], P_exact[tstep, x_zero_pos], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t10
tstep = t_step_array[2]
ax.plot(pos[x_zero_pos,1], P[tstep, x_zero_pos], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], P_exact[tstep, x_zero_pos], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t15
tstep = t_step_array[3]
ax.plot(pos[x_zero_pos,1], P[tstep, x_zero_pos], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], P_exact[tstep, x_zero_pos], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t20
tstep = t_step_array[4]
ax.plot(pos[x_zero_pos,1], P[tstep, x_zero_pos], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], P_exact[tstep, x_zero_pos], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )


ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Distance, m', ylabel='Normalized Pressure, Pa', title="Mandel's Problem: Pressure along x = 0")
fig.tight_layout()
fig.savefig('output/mandel_pressure_x_0_raw.png',dpi = 300)

fig.show()

# Generate Z Displacement Graph
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(pos[x_zero_pos,1], U[tstep, x_zero_pos,1], color=cm_numeric[-1], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], U_exact[tstep, x_zero_pos,1], color=cm_analytic[-1], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t1
tstep = t_step_array[0]
ax.plot(pos[x_zero_pos,1], U[tstep, x_zero_pos,1], color=cm_numeric[0], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], U_exact[tstep, x_zero_pos,1], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t5
tstep = t_step_array[1]
ax.plot(pos[x_zero_pos,1], U[tstep, x_zero_pos,1], color=cm_numeric[1], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], U_exact[tstep, x_zero_pos,1], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t[tstep].ravel()) )

#t10
tstep = t_step_array[2]
ax.plot(pos[x_zero_pos,1], U[tstep, x_zero_pos,1], color=cm_numeric[2], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], U_exact[tstep, x_zero_pos,1], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t15
tstep = t_step_array[3]
ax.plot(pos[x_zero_pos,1], U[tstep, x_zero_pos,1], color=cm_numeric[3], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], U_exact[tstep, x_zero_pos,1], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )

#t20
tstep = t_step_array[4]
ax.plot(pos[x_zero_pos,1], U[tstep, x_zero_pos,1], color=cm_numeric[4], label='Numerical, t = ' + np.str(t[tstep].ravel()) )
ax.plot(pos[x_zero_pos,1], U_exact[tstep, x_zero_pos,1], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t = ' + np.str(t[tstep].ravel()) )


ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Distance, m', ylabel='Normalized Displacement, m', title="Mandel's Problem: Z Displacement at x = 0")
fig.tight_layout()
fig.savefig('output/mandel_displacement_z_raw.png',dpi = 300)

fig.show()



# ==============================================================================
# Nondimensional Output
# ==============================================================================

# Generate Pressure Graph, Z = 0
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(pos_N[z_zero_pos,0], P_N[tstep, z_zero_pos], color=cm_numeric[-1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], P_exact_N[tstep, z_zero_pos], color=cm_analytic[-1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t1
tstep = t_step_array[0]
ax.plot(pos_N[z_zero_pos,0], P_N[tstep, z_zero_pos], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], P_exact_N[tstep, z_zero_pos], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t5
tstep = t_step_array[1]
ax.plot(pos_N[z_zero_pos,0], P_N[tstep, z_zero_pos], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], P_exact_N[tstep, z_zero_pos], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t10
tstep = t_step_array[2]
ax.plot(pos_N[z_zero_pos,0], P_N[tstep, z_zero_pos], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], P_exact_N[tstep, z_zero_pos], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t15
tstep = t_step_array[3]
ax.plot(pos_N[z_zero_pos,0], P_N[tstep, z_zero_pos], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], P_exact_N[tstep, z_zero_pos], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t20
tstep = t_step_array[4]
ax.plot(pos_N[z_zero_pos,0], P_N[tstep, z_zero_pos], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], P_exact_N[tstep, z_zero_pos], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )


ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Distance, x*', ylabel='Normalized Pressure, P*', title="Mandel's Problem: Normalized Pressure at z = 0")
fig.tight_layout()
fig.savefig('output/mandel_pressure_z_0.png',dpi = 300)

fig.show()

# Generate Pressure Graph, X = 0
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[-1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], color=cm_analytic[-1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t1
tstep = t_step_array[0]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t5
tstep = t_step_array[1]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t10
tstep = t_step_array[2]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t15
tstep = t_step_array[3]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t20
tstep = t_step_array[4]
ax.plot(pos_N[x_zero_pos,1], P_N[tstep, x_zero_pos], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], P_exact_N[tstep, x_zero_pos], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )


ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Distance, z*', ylabel='Normalized Pressure, P*', title="Mandel's Problem: Normalized Pressure at x = 0")
fig.tight_layout()
fig.savefig('output/mandel_pressure_x_0.png',dpi = 300)

fig.show()


# Generate X Displacement Graph
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(pos_N[z_zero_pos,0], U_N[tstep, z_zero_pos,0], color=cm_numeric[-1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], U_exact_N[tstep, z_zero_pos,0], color=cm_analytic[-1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t1
tstep = t_step_array[0]
ax.plot(pos_N[z_zero_pos,0], U_N[tstep, z_zero_pos,0], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], U_exact_N[tstep, z_zero_pos,0], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t5
tstep = t_step_array[1]
ax.plot(pos_N[z_zero_pos,0], U_N[tstep, z_zero_pos,0], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], U_exact_N[tstep, z_zero_pos,0], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t10
tstep = t_step_array[2]
ax.plot(pos_N[z_zero_pos,0], U_N[tstep, z_zero_pos,0], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], U_exact_N[tstep, z_zero_pos,0], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t15
tstep = t_step_array[3]
ax.plot(pos_N[z_zero_pos,0], U_N[tstep, z_zero_pos,0], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], U_exact_N[tstep, z_zero_pos,0], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t20
tstep = t_step_array[4]
ax.plot(pos_N[z_zero_pos,0], U_N[tstep, z_zero_pos,0], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[z_zero_pos,0], U_exact_N[tstep, z_zero_pos,0], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Distance, x*', ylabel='Normalized Displacement, U*', title="Mandel's Problem: Normalized X Displacement at z = 0")
fig.tight_layout()
fig.savefig('output/mandel_displacement_x.png',dpi = 300)

fig.show()

# Generate Z Displacement Graph
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

#t0
tstep = 0
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[-1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], color=cm_analytic[-1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t1
tstep = t_step_array[0]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[0], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], color=cm_analytic[0], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t5
tstep = t_step_array[1]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[1], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], color=cm_analytic[1], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t10
tstep = t_step_array[2]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[2], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], color=cm_analytic[2], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t15
tstep = t_step_array[3]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[3], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], color=cm_analytic[3], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

#t20
tstep = t_step_array[4]
ax.plot(pos_N[x_zero_pos,1], U_N[tstep, x_zero_pos,1], color=cm_numeric[4], label='Numerical, t* = ' + np.str(t_N[tstep].ravel()) )
ax.plot(pos_N[x_zero_pos,1], U_exact_N[tstep, x_zero_pos,1], color=cm_analytic[4], marker='^', linestyle=' ',  label='Analytical, t* = ' + np.str(t_N[tstep].ravel()) )

ax.grid()
ax.legend(loc='best')
ax.set(xlabel='Normalized Distance, z*', ylabel='Normalized Displacement, U*', title="Mandel's Problem: Normalized Z Displacement at x = 0")
fig.tight_layout()
fig.savefig('output/mandel_displacement_z.png',dpi = 300)

fig.show()





