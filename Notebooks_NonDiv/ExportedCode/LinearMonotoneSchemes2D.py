# Code automatically exported from notebook LinearMonotoneSchemes2D.ipynb# Do not modifyimport sys; sys.path.append("../..") # Allow imports from parent directory

from NumericalSchemes import Selling
from NumericalSchemes import LinearParallel as lp
from NumericalSchemes import FiniteDifferences as fd
from NumericalSchemes import AutomaticDifferentiation as ad

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg 

spid = ad.Sparse.identity
as_field=fd.as_field
LInfNorm = ad.Optimization.norm_infinity
    
def streamplot_ij(X,Y,VX,VY,subsampling=1,*varargs,**kwargs):
    def f(array): return array[::subsampling,::subsampling].T
    return plt.streamplot(f(X),f(Y),f(VX),f(VY),*varargs,**kwargs) # Transpose everything

def SchemeCentered(u,cst,mult,omega,diff,bc,h,
                   padding=0.,ret_hmax=False):
    """Discretization of a linear non-divergence form second order PDE
        cst + mult u + <omega,grad u>- tr(diff hess(u)) = 0
        Second order accurate, centered yet monotone finite differences are used for <omega,grad u>
        - bc : Dirichlet boundary conditions, u - bc = 0. Use NaNs in domain interior. 
        - h : grid scale
        - ret_hmax : return the largest grid scale for which monotony holds
        - padding : Dirichlet bc to be used outside the enclosing box.
    """
    # Decompose the tensor field
    coefs2,offsets = Selling.Decomposition(diff)
    
    # Decompose the vector field
    scals = lp.dot_VA(lp.solve_AV(diff,omega), offsets.astype(float))
    coefs1 = coefs2*scals
    if ret_hmax: return 2./LInfNorm(scals)
    
    # Compute the first and second order finite differences    
    du  = fd.DiffCentered(u,offsets,h,padding=padding)
    d2u = fd.Diff2(u,offsets,h,padding=padding)
    
    # In interior : cst + mult u + <omega,grad u>- tr(diff hess(u)) = 0
    coefs1,coefs2 = (as_field(e,u.shape) for e in (coefs1,coefs2))    
    residue = cst + mult*u +lp.dot_VV(coefs1,du) - lp.dot_VV(coefs2,d2u)
    
    # On boundary : u-bc = 0
    return ad.where(np.isnan(bc),residue,u-bc)

def SchemeUpwind(u,cst,mult,omega,diff,bc,h,padding=0.):
    """Discretization of a linear non-divergence form second order PDE
        cst + mult u + <omega,grad u>- tr(diff hess(u)) = 0
        First order accurate, upwind finite differences are used for <omega,grad u>
        - bc : Dirichlet boundary conditions, u - bc = 0. Use NaNs in domain interior. 
        - h : grid scale
        - padding : Dirichlet bc to be used outside the enclosing box.
    """
    # Decompose the tensor field
    coefs2,offsets2 = Selling.Decomposition(diff)
    omega,coefs2 = (as_field(e,u.shape) for e in (omega,coefs2))    

    # Decompose the vector field
    coefs1 = -np.abs(omega)
    basis = as_field(np.eye(len(omega)),u.shape)
    offsets1 = -np.sign(omega)*basis
    
    # Compute the first and second order finite differences    
    du  = fd.DiffUpwind(u,offsets1.astype(int),h,padding=padding)
    d2u = fd.Diff2(u,offsets2,h,padding=padding)
    
    # In interior : cst + mult u + <omega,grad u>- tr(diff hess(u)) = 0
    residue = cst + mult*u +lp.dot_VV(coefs1,du) - lp.dot_VV(coefs2,d2u)
    
    # On boundary : u-bc = 0
    return ad.where(np.isnan(bc),residue,u-bc)

