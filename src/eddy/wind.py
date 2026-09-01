import geopandas as gpd
from dataclasses import dataclass
import numpy as np

@dataclass
class Constants:
    kappa: float = 0.41  # von Karman constant
    g: float = 9.81  # gravitational acceleration (m/s^2)

def logLL(z, z0, u_star, L):
    '''Extrapolate the wind profile to a given height z (m) using the logLL function'''
    zL = z / L

    

def logL(z, z0, u_star):
    '''Extrapolate the wind profile to a given height z (m) using the logL function'''
    pass





def DeavesHarris(z, z0, u_star, h):
    '''Extrapolate the wind profile to a given height z (m) using the Deaves-Harris function'''
    pass



def alpha12(v_z2, v_z1, z2, z1):
    '''Calculate the power law exponent alpha given two wind speeds at two heights'''
    pass

def alphaPLaC(z0):
    '''Calculate the power law exponent alpha using the Power Law C method given a roughness length z0'''
    pass

def powerLaw(v0, z, z0, alpha):
    '''Extrapolate the wind profile to a given height z (m) using the power law function'''
    pass