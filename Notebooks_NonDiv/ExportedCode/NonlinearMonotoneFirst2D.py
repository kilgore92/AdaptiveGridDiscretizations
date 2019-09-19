# Code automatically exported from notebook NonlinearMonotoneFirst2D.ipynb# Do not modifyimport sys; sys.path.append("../..") # Allow imports from parent directory

from NumericalSchemes import Selling
from NumericalSchemes import LinearParallel as lp
from NumericalSchemes import FiniteDifferences as fd
from NumericalSchemes import AutomaticDifferentiation as ad

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg;
import itertools

def Gradient(u,A,h,padding=0.,decomp=None):
    """
    Approximates grad u(x), using finite differences along the axes of A.
    """
    coefs,offsets = Selling.Decomposition(A) if decomp is None else decomp
    du = fd.DiffCentered(u,offsets,h,padding=padding)
    AGrad = lp.dot_AV(offsets.astype(float),(coefs*du))
    return lp.solve_AV(A,AGrad)

def SchemeLaxFriedrichs(u,A,F,bc,h,padding=0.):
    """
    Discretization of - Tr(A(x) hess u(x)) + F(grad u(x)) - 1 = 0,
    with Dirichlet boundary conditions. The scheme is second order,
    and degenerate elliptic under suitable assumptions.
    """
    # Compute the tensor decomposition
    coefs,offsets = Selling.Decomposition(A)
    A,coefs,offsets = (fd.as_field(e,u.shape) for e in (A,coefs,offsets))
    
    # Obtain the first and second order finite differences
    grad = Gradient(u,A,h,padding=padding,decomp=(coefs,offsets))
    d2u = fd.Diff2(u,offsets,h,padding=padding)    
    
    # Numerical scheme in interior    
    residue = -lp.dot_VV(coefs,d2u) + F(grad) -1.
    
    # Boundary conditions
    return ad.where(np.isnan(bc),residue,u-bc)

# Specialization for the quadratic non-linearity
def SchemeLaxFriedrichs_Quad(u,A,omega,D,bc,h,padding=0.):
    omega,D = (fd.as_field(e,u.shape) for e in (omega,D))
    def F(g): return lp.dot_VAV(g-omega,D,g-omega)
    return SchemeLaxFriedrichs(u,A,F,bc,h,padding=padding)

