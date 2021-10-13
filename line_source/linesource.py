import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from mpl_toolkits import mplot3d
from scipy.misc import derivative
from scipy import integrate

# define some useful perpendicular distance
def r1(X):
    return X[0]**2 + (X[1] - D)**2
def r2(X):
    return X[0]**2 + (X[1] + D)**2

# first we have the displacement Green's functions for the undrained response 
def ux(X):
    term1 = X[0]/r1(X)**2
    term2 = (3-4*nu_u)*X[0]/r2(X)**2
    term3 = 4*X[1]*(X[1]+D)*X[0]/r2(X)**4
    return gov2pi*( term1 +  term2 - term3 )

def uy(X):
    term1 = (X[1]-D)/r1(X)**2
    term2 = 2*X[1]-(3-4*nu_u)*(X[1]+D)/r2(X)**2
    term3 = 4*X[1]*(X[1]+D)*X[0]/r2(X)**4
    return gov2pi*( term1 +  term2 - term3 )
    
# now we will define useful things to find the time-dependent response by convolution

# integrated erfc
def ierfc(x):
    return np.exp(-x**2)/np.sqrt(np.pi) - x*(1 - erf(x))

#define the integral using trapezoidal rule 
def intterm(xi,tt):
    argum = np.sqrt(xint**2/(4*c*t))
    denom = D**2 + (xi - xint)**2
    integrand = ierfc(argum)/denom
    return integrate.quad(integrand, xint, initial=0)

#vertical displacement at the surface (y=0)
def uy_0(X,t):
    arrayint = [intterm(xi,t*3.0e7) for xi in X[0]]
    # find the integral at each value of x
    return arrayint


#plot vertical displacement at surface as a function of x, at various times t
times = [0.001, 1, 5, 10, 20, 50, 100] #times in yrs
x1 = np.linspace(-2.0, 2.0, 100) # in km
x2 = np.linspace(0, 5.0, 10) # in km
X = np.array([x1, x2]) 
xint = np.linspace(-2.1, 2.2, 200) #make a dummy array for integration in km

# Material properties
B = 0.6 # Skempton's coefficient
nu_u = 0.33 # undrained Poisson's coeff
gamma = B*(1+nu_u)/(3*(1-nu_u))
gov2pi = 0.5*gamma/np.pi

# layer/source properties
L   = 1e4 # finite line length in m
b   = 100 # layer thickness in m
D   = 1000 # source depth in m
V_0_dot = 2.0e+6/3.0e+07 # in m^3/s
Q_0 = -V_0_dot/(L*b)
c   = 0.1 # hydraulic diffusivity in m^2/s

prefac = 2*B*(1+nu_u)*Q_0*b*D/(3*np.pi)

plt.xlabel(r"x-distance from line source at x=0")
plt.ylabel(r"Vertical displacement $u_{y}(x,0,t)$")

for t in times:
  tt = (3.0e+7)*t # convert to s
  term = np.sqrt(tt/c)
  test = prefac*term
  print(test)
  plt.plot(X[0], uy_0(X,t), label="time(yr)="+str(t))

#plt.legend()
plt.show()

#plt.legend()
plt.show()
