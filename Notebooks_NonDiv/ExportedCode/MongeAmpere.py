# Code automatically exported from notebook MongeAmpere.ipynb# Do not modifyimport sys; sys.path.append("../..") # Allow imports from parent directory

from NumericalSchemes import Selling
from NumericalSchemes import LinearParallel as lp
from NumericalSchemes import FiniteDifferences as fd
from NumericalSchemes import AutomaticDifferentiation as ad

import numpy as np
from matplotlib import pyplot as plt

as_field = fd.as_field
newton_root = ad.Optimization.newton_root
stop    = ad.Optimization.stop_default
damping = ad.Optimization.damping_default 

def SchemeNonMonotone(u,f,bc,h):
    # Compute the hessian matrix of u
    uxx = fd.Diff2(u,(1,0),h)
    uyy = fd.Diff2(u,(0,1),h)
    uxy = 0.25*(fd.Diff2(u,(1,1),h) - fd.Diff2(u,(1,-1),h))
    
    # Numerical scheme
    det = uxx*uyy-uxy**2
    residue = f - det
    
    # Boundary conditions
    return ad.where(np.isnan(bc),residue,u-bc)

def MALBR_H(d2u):
    a,b,c = ad.sort(np.maximum(0.,d2u), axis=0)
    
    # General formula, handling infinite values separately
    A,B,C = (ad.where(e==np.inf,0.,e) for e in (a,b,c))
    result = 0.5*(A*B+B*C+C*A)-0.25*(A**2+B**2+C**2)
    
    pos_inf = np.logical_or.reduce(d2u==np.inf)    
    result[pos_inf]=np.inf
    
    pos_ineq = a+b<c
    result[pos_ineq] = (A*B)[pos_ineq]
        
    return result
    
def SchemeMALBR(u,SB,f,bc,gridScale):
    # Compute the finite differences along the superbase directions
    d2u = fd.Diff2(u,SB,gridScale,padding=np.inf)
    
    # Numerical scheme
    residue = f-MALBR_H(d2u).min(axis=0)
    
    # Boundary conditions
    return ad.where(np.isnan(bc),residue,u-bc)

def InvalidMALBR(u,SB,f,bc,h):
    residue = SchemeMALBR(u,SB,f,bc,gridScale)
    interior = np.isnan(bc)
    return np.any(residue[interior]>=f/2)

def SchemeMALBR_OptInner(u,SB,gridScale,oracle=None):
    # If the active superbases are known, then take only these
    if not(oracle is None):
        SB = np.take_along_axis(SB,np.broadcast_to(oracle,SB.shape[:2]+(1,)+oracle.shape),axis=2)
                
    d2u = fd.Diff2(u,SB,gridScale,padding=np.inf)    
    # Evaluate the complex non-linear function using dense - sparse composition
    result = ad.apply(MALBR_H,d2u,shape_bound=u.shape)
    
    return ad.min_argmin(result,axis=0)

def SchemeMALBR_Opt(u,SB,f,bc,gridScale):
    
    # Evaluate using the envelope theorem
    result,_ = ad.apply(SchemeMALBR_OptInner, u,as_field(SB,u.shape),gridScale, envelope=True)
        
    # Boundary conditions
    return ad.where(np.isnan(bc), f - result, u-bc)

def ConstrainedMaximize(Q,l,m):
    dim = l.shape[0]
    if dim==1:
        return (l[0]+np.sqrt(Q[0,0]))/m[0]
    
    # Discard infinite values, handled afterwards
    pos_bad = l.min(axis=0)==-np.inf
    L = l.copy(); L[:,pos_bad]=0
    
    # Solve the quadratic equation
    A = lp.inverse(Q)    
    lAl = lp.dot_VAV(L,A,L)
    lAm = lp.dot_VAV(L,A,m)
    mAm = lp.dot_VAV(m,A,m)
    
    delta = lAm**2 - (lAl-1.)*mAm
    pos_bad = np.logical_or(pos_bad,delta<=0)
    delta[pos_bad] = 1.
    
    mu = (lAm + np.sqrt(delta))/mAm
    
    # Check the positivity
#    v = dot_AV(A,mu*m-L)
    rm_ad = np.array
    v = lp.dot_AV(rm_ad(A),rm_ad(mu)*rm_ad(m)-rm_ad(L))
    pos_bad = np.logical_or(pos_bad,np.any(v<0,axis=0))
    
    result = mu
    result[pos_bad] = -np.inf
    
    # Solve the lower dimensional sub-problems
    # We could restrict to the bad positions, and avoid repeating computations
    for i in range(dim):             
        axes = np.full((dim),True); axes[i]=False
        res = ConstrainedMaximize(Q[axes][:,axes],l[axes],m[axes])
        result = np.maximum(result,res)
    return result

def SchemeUniform(u,SB,f,bc,h):
    # Compute the finite differences along the superbase directions
    d2u = fd.Diff2(u,SB,h) #,padding=np.inf)
    
    # Generate the parameters for the low dimensional optimization problem
    Q = 0.5*np.array([[0,1,1],[1,0,1],[1,1,0]])
    dim = 2
    l = -d2u/(dim * f**(1/dim))
    m = (SB**2).sum(0)

    # Evaluate the numerical scheme
    m = as_field(m,u.shape)
    Q = as_field(Q,m.shape[1:])
    residue = ConstrainedMaximize(Q,l,m).max(axis=0)
    
    # Boundary conditions
    return ad.where(np.isnan(bc),residue,u-bc)

def SchemeUniform_OptInner(u,SB,f,h,oracle=None):
    # Use the oracle, if available, to select the active superbases only
    if not(oracle is None):
        SB = np.take_along_axis(SB,np.broadcast_to(oracle,SB.shape[:2]+(1,)+oracle.shape),axis=2)

    d2u = fd.Diff2(u,SB,gridScale) 
    
    # Generate the parameters for the low dimensional optimization problem
    Q = 0.5*np.array([[0,1,1],[1,0,1],[1,1,0]])
    dim = 2
    l = -d2u/(dim * f**(1/dim))
    m = (SB**2).sum(0)

    
    m = as_field(m,u.shape)
    Q = as_field(Q,m.shape[1:])
    
    # Evaluate the non-linear functional using dense-sparse composition
    result = ad.apply(ConstrainedMaximize, Q,l,m, shape_bound=u.shape)
    
    return ad.max_argmax(result,axis=0)

def SchemeUniform_Opt(u,SB,f,bc,h):
    
    # Evaluate the maximum over the superbases using the envelope theorem
    residue,_ = ad.apply(SchemeUniform_OptInner, u,as_field(SB,u.shape),f,h, envelope=True)
    
    return ad.where(np.isnan(bc),residue,u-bc)

