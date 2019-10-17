# Code automatically exported from notebook NonlinearMonotoneSecond2D.ipynb# Do not modifyimport sys; sys.path.append("../..") # Allow imports from parent directory

from NumericalSchemes import Selling
from NumericalSchemes import LinearParallel as lp
from NumericalSchemes import FiniteDifferences as fd
from NumericalSchemes import AutomaticDifferentiation as ad

import numpy as np
import matplotlib.pyplot as plt

def SchemeNonMonotone(u,alpha,beta,bc,h):
    # Compute the hessian matrix of u
    uxx = fd.Diff2(u,(1,0),h)
    uyy = fd.Diff2(u,(0,1),h)
    uxy = 0.25*(fd.Diff2(u,(1,1),h) - fd.Diff2(u,(1,-1),h))
    
    # Compute the eigenvalues
    htr = 0.5*(uxx+uyy)
    det = uxx*uyy-uxy**2
    
    delta = htr**2-det
    lambda_max = htr+np.sqrt(delta)
    lambda_min = htr-np.sqrt(delta)
    
    # Numerical scheme
    residue = beta - alpha*lambda_max - lambda_min
    
    # Boundary conditions
    return ad.where(np.isnan(bc),residue,u-bc)

def SchemeSampling(u,diffs,beta,bc,gridScale):
    # Tensor decomposition 
    coefs,offsets = Selling.Decomposition(diffs)
    
    # Numerical scheme 
    coefs = as_field(coefs,u.shape)
    residue = beta - (coefs*fd.Diff2(u,offsets,gridScale)).sum(0).min(0)
    
    # Boundary conditions
    return ad.where(np.isnan(bc),residue,u-bc)

def Diff(alpha,theta):
    e0 = np.array((np.cos(theta),np.sin(theta)))
    e1 = np.array((-np.sin(theta),np.cos(theta)))
    if isinstance(alpha,np.ndarray): 
        e0,e1 = (as_field(e,alpha.shape) for e in (e0,e1))
    return alpha*lp.outer_self(e0) + lp.outer_self(e1)

def SchemeSampling_OptInner(u,diffs,gridScale,oracle=None):
    # Select the active tensors, if they are known
    if not(oracle is None):
        diffs = np.take_along_axis(diffs, np.broadcast_to(oracle,diffs.shape[:2]+(1,)+oracle.shape),axis=2)
    
    print("Has AD information :", ad.is_ad(u), ". Number active tensors per point :", diffs.shape[2])
    
    # Tensor decomposition 
    coefs,offsets = Selling.Decomposition(diffs)
    
    # Return the minimal value, and the minimizing index
    return ad.min_argmin( (coefs*fd.Diff2(u,offsets,gridScale)).sum(0), axis=0)

def SchemeSampling_Opt(u,diffs,beta,bc,gridScale):
    # Evaluate the operator using the envelope theorem
    result,_ = ad.apply(SchemeSampling_OptInner, u,as_field(diffs,u.shape),gridScale, envelope=True)
        
    # Boundary conditions
    return ad.where(np.isnan(bc), beta - result, u-bc)

def MakeD(alpha):
    return np.moveaxis(0.5*np.array([
        (alpha+1)*np.array([[1,0],[0,1]]),
        (alpha-1)*np.array([[1,0],[0,-1]]),
        (alpha-1)*np.array([[0,1],[1,0]])
    ]), 0,-1)

def NextAngleAndSuperbase(theta,sb,D):
    pairs = np.stack([(1,2), (2,0), (0,1)],axis=1)
    scals = lp.dot_VAV(np.expand_dims(sb[:,pairs[0]],axis=1), 
                       np.expand_dims(D,axis=-1), np.expand_dims(sb[:,pairs[1]],axis=1))
    phi = np.arctan2(scals[2],scals[1])
    cst = -scals[0]/np.sqrt(scals[1]**2+scals[2]**2)
    theta_max = np.pi*np.ones(3)
    mask = cst<1
    theta_max[mask] = (phi[mask]-np.arccos(cst[mask]))/2
    theta_max[theta_max<=0] += np.pi
    theta_max[theta_max<=theta] = np.pi
    k = np.argmin(theta_max)
    i,j = (k+1)%3,(k+2)%3
    return (theta_max[k],np.stack([sb[:,i],-sb[:,j],sb[:,j]-sb[:,i]],axis=1))

def AnglesAndSuperbases(D,maxiter=200):
    sb = Selling.CanonicalSuperbase(2).astype(int)
    thetas=[]
    superbases=[]
    theta=0
    for i in range(maxiter):
        thetas.append(theta)
        if(theta>=np.pi): break
        superbases.append(sb)
        theta,sb = NextAngleAndSuperbase(theta,sb,D)
    return np.array(thetas), np.stack(superbases,axis=2)

def MinimizeTrace(u,alpha,h):
    # Compute the tensor decompositions
    D=MakeD(alpha)
    theta,sb = AnglesAndSuperbases(D)
    theta = np.array([theta[:-1],theta[1:]])
    
    # Compute the second order differences in the direction orthogonal to the superbase
    sb_rotated = np.array([-sb[1],sb[0]])
    d2u = fd.Diff2(u,sb_rotated,h)
    
    # Compute the coefficients of the tensor decompositions
    sb1,sb2 = np.roll(sb,1,axis=1), np.roll(sb,2,axis=1)
    sb1,sb2 = (e.reshape( (2,3,1)+sb.shape[2:]) for e in (sb1,sb2))
    D = D.reshape((2,2,1,3,1)+D.shape[3:])
    # Axes of D are space,space,index of superbase element, index of D, index of superbase, and possibly shape of u
    scals = lp.dot_VAV(sb1,D,sb2)

    # Compute the coefficients of the trigonometric polynomial
    scals,theta = (as_field(e,u.shape) for e in (scals,theta))
    coefs = -(scals*np.expand_dims(d2u,axis=1)).sum(axis=0)
    
    # Optimality condition for the trigonometric polynomial in the interior
    value = coefs[0] - np.sqrt(coefs[1]**2+coefs[2]**2)
    coefs_ = np.array(coefs) # removed AD information
    angle = np.arctan2(-coefs_[2],-coefs_[1])/2.
    angle[angle<0]+=np.pi
    
    # Boundary conditions for the trigonometric polynomial minimization
    mask = np.logical_not(np.logical_and(theta[0]<=angle,angle<=theta[1]))
    t,c = theta[:,mask],coefs[:,mask]
    value[mask],amin_t = ad.min_argmin(c[0]+c[1]*np.cos(2*t)+c[2]*np.sin(2*t),axis=0)
        
    # Minimize over superbases
    value,amin_sb = ad.min_argmin(value,axis=0)
    
    # Record the optimal angles for future use
    angle[mask]=np.take_along_axis(t,np.expand_dims(amin_t,axis=0),axis=0).squeeze(axis=0) # Min over bc
    angle = np.take_along_axis(angle,np.expand_dims(amin_sb,axis=0),axis=0) # Min over superbases

    return value,angle

def SchemeConsistent(u,alpha,beta,bc,h):
    value,_ = MinimizeTrace(u,alpha,h)
    residue = beta - value
    return ad.where(np.isnan(bc),residue,u-bc)

def MinimizeTrace_Opt(u,alpha,h,oracle=None):
    if oracle is None:  return MinimizeTrace(u,alpha,h)
    
    # The oracle contains the optimal angles
    diffs=Diff(alpha,oracle.squeeze(axis=0))
    coefs,sb = Selling.Decomposition(diffs)
    value = (coefs*fd.Diff2(u,sb,h)).sum(axis=0)
    return value,oracle
    

def SchemeConsistent_Opt(u,alpha,beta,bc,h):
    value,_ = ad.apply(MinimizeTrace_Opt,u,alpha,h,envelope=True)
    residue = beta - value
    return ad.where(np.isnan(bc),residue,u-bc)

