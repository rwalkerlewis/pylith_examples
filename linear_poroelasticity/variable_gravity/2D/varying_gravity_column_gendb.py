#!/usr/bin/env python
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
#
# @file tests/fullscale/linearporoelasticity/mandel/mandel_gendb.py
#
# @brief Python script to generate spatial database with displacement
# boundary conditions for the mandel test.

import numpy


class GenerateDB(object):
    """
    Python object to generate spatial database with displacement
    boundary conditions for the axial displacement test.
    """

    def run(self):
        """
        Generate the database.
        """
        # Domain
        x = numpy.arange(0.0, 10.1, 0.1)
        y = numpy.arange(0.0, 1.01, 0.01)
        npts = x.shape[0]

        xx = x * numpy.ones((npts, 1), dtype=numpy.float64)
        yy = y * numpy.ones((npts, 1), dtype=numpy.float64)
        xy = numpy.zeros((npts**2, 2), dtype=numpy.float64)
        xy[:, 0] = numpy.ravel(xx)
        xy[:, 1] = numpy.ravel(numpy.transpose(yy))

        from mandel_soln import AnalyticalSoln
        soln = AnalyticalSoln()
        disp = soln.initial_displacement(xy)
        pres = soln.initial_pressure(xy)
        trace_strain = soln.initial_trace_strain(xy)

        from spatialdata.geocoords.CSCart import CSCart
        cs = CSCart()
        cs.inventory.spaceDim = 2
        cs._configure()
        data = {
                'x' : x,
                'y' : y,
                'points': xy,
                'coordsys': cs,
                'data_dim': 2,
                'values': [{'name': "gravity_field_x",
                            'units': "m / s**2",
                            'data': numpy.ravel(gravity_field[0, :, 0])},
                           {'name': "gravity_field_y",
                            'units': "m / s**2",
                            'data': numpy.ravel(gravity_field[0, :, 1])},
                           {'name': "gravity_field_z",
                            'units': "m / s**2",
                            'data': numpy.ravel(gravity_field[0, :, 2])}]}

        from spatialdata.spatialdb.SimpleGridAscii import SimpleGridAscii
        io = SimpleGridAscii()
        io.inventory.filename = "varying_gravity_column.spatialdb"
        io._configure()
        io.write(data)
        return


# ======================================================================
if __name__ == "__main__":
    GenerateDB().run()


# End of file
