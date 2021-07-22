#!/usr/bin/env nemesis
#
# ----------------------------------------------------------------------
#
# Brad T. Aagaard, U.S. Geological Survey
# Charles A. Williams, GNS Science
# Matthew G. Knepley, University of Chicago
#
# This code was developed as part of the Computational Infrastructure
# for Geodynamics (http://geodynamics.org).
#
# Copyright (c) 2010-2016 University of California, Davis
#
# See COPYING for license information.
#
# ----------------------------------------------------------------------

import numpy 
from netCDF4 import Dataset

# mesh = Dataset('mesh_hex.exo','r')

# SPE 10 Parameters
nx = 60
ny = 220
nz = 85

dx = 20.
dy = 10.
dz = 2.

phi = numpy.loadtxt('spe_phi.dat').ravel()
phi = phi.reshape([nx,ny,nz])

perm = numpy.loadtxt('spe_perm.dat').ravel() * 9.869233e-16 

k_x = perm[0: nx*ny*nz].reshape([nx,ny,nz])
k_y = perm[nx*ny*nz:2*nx*ny*nz].reshape([nx,ny,nz])
k_z = perm[2*nx*ny*nz:3*nx*ny*nz].reshape([nx,ny,nz])

x_n = numpy.arange(0, nx*dx + dx, dx)
y_n = numpy.arange(0, ny*dy + dy, dy)
z_n = numpy.arange(0, nz*dz + dy, dz)

class GenerateDB(object):
    """Python object to generate spatial database with displacement
    boundary conditions for the axial displacement test.
    """

    def run(self):
        """Generate the database.
        """
        # Domain, centroid
        x_c = numpy.arange(0, nx*dx, dx) + dx/2
        y_c = numpy.arange(0, ny*dy, dy) + dy/2
        z_c = numpy.arange(0, nz*dz, dz) + dz/2        
        x_c_3, y_c_3, z_c_3 = numpy.meshgrid(x_c, y_c, z_c)

        xyz_c = numpy.zeros((nx*ny*nz, 3), dtype=numpy.float64)
        xyz_c[:, 0] = x_c_3.ravel()
        xyz_c[:, 1] = y_c_3.ravel()
        xyz_c[:, 2] = z_c_3.ravel()

        from spatialdata.geocoords.CSCart import CSCart
        cs = CSCart()
        cs.inventory.spaceDim = 3
        cs._configure()
        data = {
            "x": x_c,
            "y": y_c,
            "z": z_c,
            "points": xyz_c,
            "coordsys": cs,
            "data_dim": 3,
            "values": [
                {"name": "porosity",
                 "units": "none",
                 "data": numpy.ravel(phi)},
                {"name": "permeability_xx",
                 "units": "m*m",
                 "data": numpy.ravel(k_x)},
                {"name": "permeability_yy",
                 "units": "m*m",
                 "data": numpy.ravel(k_y)},                 
                {"name": "permeability_zz",
                 "units": "m*m",
                 "data": numpy.ravel(k_z)},
                {"name": "permeability_xy",
                 "units": "m*m",
                 "data": numpy.zeros(k_x.size)},
                {"name": "permeability_yz",
                 "units": "m*m",
                 "data": numpy.zeros(k_y.size)},                                  
                {"name": "permeability_xz",
                 "units": "m*m",
                 "data": numpy.zeros(k_z.size)},
                {"name": "solid_density",
                 "units": "kg/m**3",
                 "data": numpy.ones(nx*ny*nz)},
                {"name": "fluid_density",
                 "units": "kg/m**3",
                 "data": numpy.ones(nx*ny*nz)},
                {"name": "fluid_viscosity",
                 "units": "Pa*s",
                 "data": numpy.ones(nx*ny*nz)},
                {"name": "shear_modulus",
                 "units": "Pa",
                 "data": numpy.ones(nx*ny*nz)},
                {"name": "drained_bulk_modulus",
                 "units": "Pa",
                 "data": numpy.ones(nx*ny*nz)},
                {"name": "solid_bulk_modulus",
                 "units": "Pa",
                 "data": numpy.ones(nx*ny*nz)},
                {"name": "fluid_bulk_modulus",
                 "units": "Pa",                 
                 "data": numpy.ones(nx*ny*nz)},
                {"name": "biot_coefficient",
                 "units": "none",
                 "data": numpy.ones(nx*ny*nz)},
            ]}

        from spatialdata.spatialdb.SimpleGridAscii import SimpleGridAscii
        io = SimpleGridAscii()
        io.inventory.filename = "SPE10_parameters.spatialdb"
        io._configure()
        io.write(data)
        return

# ======================================================================
if __name__ == "__main__":
    GenerateDB().run()


# End of file
