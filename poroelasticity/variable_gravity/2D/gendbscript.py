#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset

# Load grid
filename = 'mesh_quad.exo'
quad = Dataset('mesh_quad.exo','r')  
X = np.array(quad.variables['coordx'][:])
Y = np.array(quad.variables['coordy'][:])
xy = np.array([X[:], Y[:]]).T
connect = quad.variables['connect1']

# Parameters
decimal_precision = 3
overlap = 0.1

x_range = np.unique(np.round(X,decimal_precision))
y_range = np.unique(np.round(Y,decimal_precision))

xmin = x_range.min()
xmax = x_range.max()
ymin = y_range.min()
ymax = y_range.max()

npts_x = x_range.size
npts_y = y_range.size

# Domain
#x_dom = np.arange(0.0, 10.1, 0.1)
#y_dom = np.arange(0.0, 1.01, 0.01)
x_dom = np.linspace(xmin, xmax, npts_x)
y_dom = np.linspace(ymin, ymax, npts_y)

npts_dom_x = x_dom.shape[0]
npts_dom_y = y_dom.shape[0]

xx_dom = x_dom * np.ones((npts_dom_x, 1), dtype=np.float64)
yy_dom = y_dom * np.ones((npts_dom_y, 1), dtype=np.float64)
xy_dom = np.zeros((npts_dom_x*npts_dom_y, 2), dtype=np.float64)
xy_dom[:, 0] = np.ravel(xx_dom)
xy_dom[:, 1] = np.ravel(np.transpose(yy_dom))

# Write field

(npts, dim) = xy_dom.shape

gravity_field = np.zeros([1,npts,3], dtype=np.float64)
gravity_field[0,:,0] = 0.0
gravity_field[0,:,1] = -9.80665
gravity_field[0,:,2] = 0.0

from spatialdata.geocoords.CSCart import CSCart
cs = CSCart()
cs.inventory.spaceDim = dim
cs._configure()
data = {
        'x' : x_dom,
        'y' : y_dom,
        'points': xy_dom,
        'coordsys': cs,
        'data_dim': dim,
        'values': [{'name': "gravity_field_x",
                    'units': "m / s**2",
                    'data': np.ravel(gravity_field[0, :, 0])},
                   {'name': "gravity_field_y",
                    'units': "m / s**2",
                    'data': np.ravel(gravity_field[0, :, 1])},
                   {'name': "gravity_field_z",
                    'units': "m / s**2",
                    'data': np.ravel(gravity_field[0, :, 2])}]}

from spatialdata.spatialdb.SimpleGridAscii import SimpleGridAscii
io = SimpleGridAscii()
io.inventory.filename = "varying_gravity_column.spatialdb"
io._configure()
io.write(data)

        
