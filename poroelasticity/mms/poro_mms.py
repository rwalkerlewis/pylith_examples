#!/usr/bin/env python

# MMS test checks for linear poroelasticity
import numpy as np
import sympy as sp

x, y, z, t, p = sp.symbols('x y z t p')



coord_one = np.array([[-4.0e+3, -4.0e+3],
                        [-4.0e+3,  4.0e+3],
                        [ 4.0e+3, -4.0e+3],
                        [ 4.0e+3,  4.0e+3]])

t_c = 0

x_c = coord_one[:,0].copy() / 1000
y_c = coord_one[:,1].copy() / 1000

u_x = x_c**2
u_y = y_c**2 - 2*x_c*y_c
e_v = 2*y_c
p_c = (x_c + y_c)*t_c


# One Block Coordinates
pos_OB = np.array([ [-0.64288254347276719, -0.68989794855663567],
                    [-0.84993777955478378, 0.28989794855663553],
                    [0.3327804920294028, -0.68989794855663567],
                    [-0.43996016900185181, 0.28989794855663553]])
                    
      
u_x_OB = pos_OB[:,0]**2
u_y_OB = pos_OB[:,1]**2 - 2*pos_OB[:,0]*pos_OB[:,1]                    
e_v_OB = 2*pos_OB[:,1]


quad_coord = np.array([ [-4.0,  -4.0],
                        [-4.0,  -2.0],
                        [-4.0,  +2.0],
                        [-4.0,  +4.0],
                        [-2.0,  -4.0],
                        [-2.0,  -2.0],
                        [-2.0,  +2.0],
                        [-2.0,  +4.0],
                        [+2.0,  -4.0],
                        [+2.0,  -2.0],
                        [+2.0,  +2.0],
                        [+2.0,  +4.0],
                        [+4.0,  -4.0],
                        [+4.0,  -2.0]])

u_x_quad = quad_coord[:,0]**2
u_y_quad = quad_coord[:,1]**2 - 2*quad_coord[:,0]*quad_coord[:,1]                    
e_v_quad = 2*quad_coord[:,1]     


