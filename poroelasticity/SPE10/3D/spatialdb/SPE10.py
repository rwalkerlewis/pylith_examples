

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

mesh = Dataset('../mesh/mesh_hex.exo','r')

# SPE 10 Parameters

nx = 60
ny = 220
nz = 85

dx = 20.
dy = 10.
dz = 2.

x_c = np.arange(0, nx*dx, dx) + dx/2
y_c = np.arange(0, ny*dy, dy) + dy/2
z_c = np.arange(0, nz*dz, dz) + dz/2

x_c_3, y_c_3, z_c_3 = np.meshgrid(x_c, y_c, z_c)

xyz_c = np.zeros((nx*ny*nz, 3), dtype=np.float64)
xyz_c[:, 0] = x_c_3.ravel()
xyz_c[:, 1] = y_c_3.ravel()
xyz_c[:, 2] = z_c_3.ravel()

x_n = np.arange(0, nx*dx + dx, dx)
y_n = np.arange(0, ny*dy + dy, dy)
z_n = np.arange(0, nz*dz + dy, dz)

x_n_3, y_n_3, z_n_3 = np.meshgrid(x_n, y_n, z_n)

phi = np.loadtxt('spe_phi.dat').ravel()
phi = phi.reshape([nz,ny,nx])

perm = np.loadtxt('spe_perm.dat').ravel()

k_x = perm[0: nx*ny*nz].reshape([nz,ny,nx])
k_y = perm[nx*ny*nz:2*nx*ny*nz].reshape([nz,ny,nx])
k_z = perm[2*nx*ny*nz:3*nx*ny*nz].reshape([nz,ny,nx])



