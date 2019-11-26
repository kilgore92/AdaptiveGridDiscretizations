# Code automatically exported from notebook Notebooks_NonDiv/NonlinearMonotoneFirst2D.ipynb# Do not modifyimport sys; sys.path.append("../..") # Allow imports from parent directory

from agd import Selling
from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad
from agd import Domain

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg;
import itertools

def Gradient(u,A,bc,decomp=None):
    """
    Approximates grad u(x), using finite differences along the axes of A.
    """
    coefs,offsets = Selling.Decomposition(A) if decomp is None else decomp
    du = bc.DiffCentered(u,offsets) 
    AGrad = lp.dot_AV(offsets.astype(float),(coefs*du)) # Approximates A * grad u
    return lp.solve_AV(A,AGrad) # Approximates A^{-1} (A * grad u) = grad u

def SchemeLaxFriedrichs(u,A,F,bc):
    """
    Discretization of - Tr(A(x) hess u(x)) + F(grad u(x)) - 1 = 0,
    with Dirichlet boundary conditions. The scheme is second order,
    and degenerate elliptic under suitable assumptions.
    """
    # Compute the tensor decomposition
    coefs,offsets = Selling.Decomposition(A)
    A,coefs,offsets = (bc.as_field(e) for e in (A,coefs,offsets))
    
    # Obtain the first and second order finite differences
    grad = Gradient(u,A,bc,decomp=(coefs,offsets))
    d2u = bc.Diff2(u,offsets)    
    
    # Numerical scheme in interior    
    residue = -lp.dot_VV(coefs,d2u) + F(grad) -1.
    
    # Placeholders outside domain
    return ad.where(bc.interior,residue,u-bc.grid_values)

# Specialization for the quadratic non-linearity
def SchemeLaxFriedrichs_Quad(u,A,omega,D,bc):
    omega,D = (bc.as_field(e) for e in (omega,D))
    def F(g): return lp.dot_VAV(g-omega,D,g-omega)
    return SchemeLaxFriedrichs(u,A,F,bc)

