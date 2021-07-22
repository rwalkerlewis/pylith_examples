#!/usr/bin/env/ python

import numpy as np
import matplotlib.pyplot as plt
import h5py



# ==============================================================================

case = '45_deg_0.5m'
path = '../' + case + '/output/fault_diffusion_quad-domain.h5'

f = h5py.File(path,'r')

t = f['time'][:]
t = t.ravel()

U = f['vertex_fields/displacement'][:]
P = f['vertex_fields/pressure'][:]
S = f['vertex_fields/trace_strain'][:]

pos = f['geometry/vertices'][:]





