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
import scipy.ndimage

# mesh = Dataset('mesh_hex.exo','r')

# SPE 10 Parameters
nx = 60
ny = 220
nz = 85

dx = 20. / 0.3048
dy = 10. / 0.3048
dz = 2. / 0.3048

# Select which layer in xy plane
zval = 0

phi = numpy.loadtxt('spe_phi.dat').ravel()
phi = phi.reshape([nz,ny,nx])

perm = numpy.loadtxt('spe_perm.dat').ravel() * 9.869233e-16 

k_x = perm[0: nz*ny*nx].reshape([nz,ny,nx])
k_y = perm[nz*ny*nx:2*nz*ny*nx].reshape([nz,ny,nx])

k_x = k_x[zval,:,:]
k_y = k_y[zval,:,:]
phi = phi[zval,:,:]

# Checkerboard
factor = 20
nx_check = numpy.int32(nx/factor)
ny_check = numpy.int32(ny/factor)

check = numpy.indices([ny_check,nx_check]).sum(axis=0) % 2
check = scipy.ndimage.zoom(check, factor, order=0)

# Two Dimensional Values

fluid_density = (check + 1)*800
solid_density = check*1000 + 1



x_n = numpy.arange(0, nx*dx + dx, dx)
y_n = numpy.arange(0, ny*dy + dy, dy)

class GenerateDB(object):


    def run(self):
        """Generate the database.
        """
        
        # Domain, centroid
        x = numpy.arange(0, nx*dx, dx) + dx/2
        y = numpy.arange(0, ny*dy, dy) + dy/2
        x2, y2 = numpy.meshgrid(x,y)
        npts_x = x.shape[0]
        npts_y = y.shape[0]
        
        # xx = x.reshape([1, x.shape[0]]) * numpy.ones([ny ,1])
        # yy = y.reshape([y.shape[0], 1]) * numpy.ones([1, nx])
        xy = numpy.zeros((npts_x*npts_y, 2), dtype=numpy.float64)
        # xy[:, 0] = numpy.ravel(xx)
        # xy[:, 1] = numpy.ravel(numpy.transpose(yy))
        xy[:, 0] = x2.ravel()
        xy[:, 1] = y2.ravel()        
        
        from spatialdata.geocoords.CSCart import CSCart
        cs = CSCart()
        cs.inventory.spaceDim = 2
        cs._configure()
        data = {
            "x": x,
            "y": y,
            "points": xy,
            "coordsys": cs,
            "data_dim": 2,
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
                 "data": numpy.zeros(nx*ny)},
                {"name": "permeability_xy",
                 "units": "m*m",
                 "data": numpy.zeros(nx*ny)},                                  
                {"name": "solid_density",
                 "units": "kg/m**3",
                 "data": numpy.ravel(solid_density)},
                {"name": "fluid_density",
                 "units": "kg/m**3",
                 "data": numpy.ravel(fluid_density)},
                {"name": "fluid_viscosity",
                 "units": "Pa*s",
                 "data": numpy.ones(nx*ny)},
                {"name": "shear_modulus",
                 "units": "Pa",
                 "data": numpy.ones(nx*ny)},
                {"name": "drained_bulk_modulus",
                 "units": "Pa",
                 "data": numpy.ones(nx*ny)},
                {"name": "solid_bulk_modulus",
                 "units": "Pa",
                 "data": numpy.ones(nx*ny)},
                {"name": "fluid_bulk_modulus",
                 "units": "Pa",                 
                 "data": numpy.ones(nx*ny)},
                {"name": "biot_coefficient",
                 "units": "none",
                 "data": numpy.ones(nx*ny)},
            ]}

        from spatialdata.spatialdb.SimpleIOAscii import createWriter
        io = createWriter("SPE10_parameters.spatialdb")
        io.write(data)        
        return

# ======================================================================
if __name__ == "__main__":
    GenerateDB().run()


# End of file
