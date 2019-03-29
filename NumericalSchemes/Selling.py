# This file implements Selling's algorithm in dimension two and three,
# which is used to construct tensor decompositions
# It performs this on parallel on a family of matrices

import numpy as np
from itertools import cycle
from LinearP import dotP_VV, dotP_AV, perpP, crossP

# -------- Dimension based dispatch -----

def DecompP(m):
    dim = m.shape[0]
    if m.shape[1]!=dim:
        raise ValueError('DecompP error : non square matrix')
    if   dim==1: return DecompP1(m)    
    elif dim==2: return DecompP2(m)
    elif dim==3: return DecompP3(m)
    else: raise ValueError('DecompP error : unsupported dimension')

# ------- One dimensional variant (trivial) ------

def DecompP1(m):
    bounds = m.shape[2:]
    
    coefs = m
    coefs.reshape(bounds)
    
    offsets = np.ones( (1,1,) + bounds)
    return coefs, offsets    

# ------- Two dimensional variant ------

# We do everyone in parallel, without selection or early abort
def SellingP2(m,b,iterMax=20):
    iterReducedMax = 3
    iter=0
    iterReduced = 0
    sigma = cycle([(0,1,2),(1,2,0),(2,0,1)])
    while iterReduced<iterReducedMax and iter<iterMax:
        # Increment iterators
        iter += 1
        iterReduced += 1
        (i,j,k) = next(sigma)
        
        # Test for a positive angle, and modify superbase if necessary
        acute = dotP_VV(b[:,i],dotP_AV(m,b[:,j])) > 0
        if np.any(acute):
            b[:,k,acute] = b[:,i,acute]-b[:,j,acute]
            b[:,i,acute] = -b[:,i,acute]
            iterReduced=0
    
    return iterReduced==iterReducedMax
    
# Produce the matrix decomposition
def DecompP2(m):
    bounds = m.shape[2:]
    if m.shape!=(2,2,)+bounds:
        raise ValueError("DecompP2 error: Incompatible dimensions")
        
    b=np.zeros((2,3,)+bounds)
    b[:,0]=-1
    b[0,1]= 1
    b[1,2]= 1
    
    if not SellingP2(m,b):
        raise ValueError('DecompP2 error: Selling algorithm unterminated')
    
    coef=np.zeros((3,)+bounds)
    for (i,j,k) in [(0,1,2),(1,2,0),(2,0,1)]:
        coef[i] = -dotP_VV(b[:,j], dotP_AV(m, b[:,k]) )
    
    return coef,perpP(b)


# ------- Three dimensional variant -------

# We do everyone in parallel, without selection or early abort
def SellingP3(m,b,iterMax=30):
    iterReducedMax = 6
    iter=0
    iterReduced = 0
    sigma = cycle([(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)])
    while iterReduced<iterReducedMax and iter<iterMax:
        # Increment iterators
        iter += 1
        iterReduced += 1
        (i,j,k,l) = next(sigma)
        
        # Test for a positive angle, and modify superbase if necessary
        acute = dotP_VV(b[:,i],dotP_AV(m,b[:,j])) > 0
        if np.any(acute):
            b[:,k,acute] += b[:,i,acute]
            b[:,l,acute] += b[:,i,acute]
            b[:,i,acute] = -b[:,i,acute]
            iterReduced=0
    
    return iterReduced==iterReducedMax
    
def DecompP3(m):
    bounds = m.shape[2:]
    b=np.zeros((3,4,)+bounds)
    b[:,0]=-1
    b[0,1]= 1
    b[1,2]= 1
    b[2,3]= 1
    
    if not SellingP3(m,b):
        raise ValueError('DecompP3 error: Selling algorithm unterminated')
    
    coef=np.zeros((6,)+bounds)
    offset=np.zeros((3,6,)+bounds)
    for iter,(i,j,k,l) in zip(range(6),
    [(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)]):
        coef[iter] = -dotP_VV(b[:,i], dotP_AV(m, b[:,j]) )
        offset[:,iter] = crossP(b[:,k], b[:,l])
        
    return coef,offset