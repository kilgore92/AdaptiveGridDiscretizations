# This file implements Selling's algorithm in dimension two and three,
# which is used to construct tensor decompositions
# It performs this on parallel on a family of matrices

import numpy as np
from itertools import cycle
from NumericalSchemes.LinearParallel import dot_VV, dot_AV, perp, cross

iterMax2 = 100 #20
iterMax3 = 100 #30

# -------- Dimension based dispatch -----

def Decomposition(m):
    """
         Use Selling's algorithm to decompose a tensor

        input : symmetric positive definite tensor, d<=3
        output : coefficients, offsets
    """
    dim = m.shape[0]
    if m.shape[1]!=dim:
        raise ValueError('Selling.Decomposition error : non square matrix')
    if   dim==1: return Decomposition1(m)    
    elif dim==2: return Decomposition2(m)
    elif dim==3: return Decomposition3(m)
    else: raise ValueError('Selling.Decomposition error : unsupported dimension')


def GatherByOffset(T,Coefs,Offsets):
    """
        Get the coefficient of a each offset
    """
    TimeCoef = {};
    for (i,t) in enumerate(T):
        coefs = Coefs[:,i]
        offsets = Offsets[:,:,i]
        for (j,c) in enumerate(coefs):
            offset = tuple(offsets[:,j].astype(int))
            offset_m = tuple(-offsets[:,j].astype(int))
            if offset<offset_m:
                offset=offset_m
            if offset in TimeCoef:
                TimeCoef[offset][0].append(t)
                TimeCoef[offset][1].append(c)
            else:
                TimeCoef[offset] = ([t],[c])
    return TimeCoef


def CanonicalSuperbase(d, bounds = tuple()):
    b=np.zeros((d,d+1,)+bounds)
    b[:,0]=-1
    for i in range(d):
        b[i,i+1]=1
    return b

# ------- One dimensional variant (trivial) ------

def Decomposition1(m):
    bounds = m.shape[2:]
    
    coefs = m
    coefs.reshape(bounds)
    
    offsets = np.ones( (1,1,) + bounds)
    return coefs, offsets    

# ------- Two dimensional variant ------

# We do everyone in parallel, without selection or early abort
def ObtuseSuperbase2(m,b):
    """
        Use Selling's algorithm to compute an obtuse superbase.

        input : symmetric positive definite matrix m, dim=2
        input/output : superbase b (must be valid at startup)
        
        module variable : iterMax2, max number of iterations

        output : wether the algorithm succeeded
    """
    iterReducedMax = 3
    iter=0
    iterReduced = 0
    sigma = cycle([(0,1,2),(1,2,0),(2,0,1)])
    while iterReduced<iterReducedMax and iter<iterMax2:
        # Increment iterators
        iter += 1
        iterReduced += 1
        (i,j,k) = next(sigma)
        
        # Test for a positive angle, and modify superbase if necessary
        acute = dot_VV(b[:,i],dot_AV(m,b[:,j])) > 0
        if np.any(acute):
            b[:,k,acute] = b[:,i,acute]-b[:,j,acute]
            b[:,i,acute] = -b[:,i,acute]
            iterReduced=0
    
    return iterReduced==iterReducedMax
    
# Produce the matrix decomposition
def Decomposition2(m):
    """
        Use Selling's algorithm to decompose a tensor

        input : symmetric positive definite tensor 
        output : coefficients, offsets
    """
    bounds = m.shape[2:]
    if m.shape!=(2,2,)+bounds:
        raise ValueError("Selling.Decomposition2 error: Incompatible dimensions")
        
    b=CanonicalSuperbase(2,bounds)

    if not ObtuseSuperbase2(m,b):
        raise ValueError('Selling.Decomposition2 error: Selling algorithm unterminated')
    
    coef=np.zeros((3,)+bounds)
    for (i,j,k) in [(0,1,2),(1,2,0),(2,0,1)]:
        coef[i] = -dot_VV(b[:,j], dot_AV(m, b[:,k]) )
    
    return coef,perp(b)


# ------- Three dimensional variant -------

# We do everyone in parallel, without selection or early abort
def ObtuseSuperbase3(m,b):
    """
        Use Selling's algorithm to compute an obtuse superbase.

        input : symmetric positive definite matrix m, dim=3
        input/output : superbase b (must be valid at startup)
        
        module variable : iterMax3, max number of iterations

        output : wether the algorithm succeeded
    """
    iterReducedMax = 6
    iter=0
    iterReduced = 0
    sigma = cycle([(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)])
    while iterReduced<iterReducedMax and iter<iterMax3:
        # Increment iterators
        iter += 1
        iterReduced += 1
        (i,j,k,l) = next(sigma)
        
        # Test for a positive angle, and modify superbase if necessary
        acute = dot_VV(b[:,i],dot_AV(m,b[:,j])) > 0
        if np.any(acute):
            b[:,k,acute] += b[:,i,acute]
            b[:,l,acute] += b[:,i,acute]
            b[:,i,acute] = -b[:,i,acute]
            iterReduced=0
    
    return iterReduced==iterReducedMax
    
def Decomposition3(m):
    """
        Use Selling's algorithm to decompose a tensor

        input : symmetric positive definite tensor, d=3
        output : coefficients, offsets
    """
    bounds = m.shape[2:]
    
    if m.shape!=(3,3,)+bounds:
        raise ValueError("Selling.Decomposition3 error: Incompatible dimensions")

    b=CanonicalSuperbase(3,bounds)

    if not ObtuseSuperbase3(m,b):
        raise ValueError('DecompP3 error: Selling algorithm unterminated')
    
    coef=np.zeros((6,)+bounds)
    offset=np.zeros((3,6,)+bounds)
    for iter,(i,j,k,l) in zip(range(6),
    [(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)]):
        coef[iter] = -dot_VV(b[:,i], dot_AV(m, b[:,j]) )
        offset[:,iter] = cross(b[:,k], b[:,l])
        
    return coef,offset