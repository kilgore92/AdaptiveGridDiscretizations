import Selling
import numpy as np
import scipy.sparse as sp
import LinearP as LP

diff = np.zeros((3,3,2,2))
diff[:,:,0,0]=1
diff[:,:,1,1]=1

"""
 Constructs a linear operator sparse matrix, given as input 
- an array of psd matrices, denoted diff
- an array of vectors (optionnal), denoted omega
- an array of scalars (optionnal),

additional parameters
- a grid scale
- boundary conditions, possibly axis by axis 
    ('Periodic', 'Reflected', 'Neumann', 'Dirichlet') 
- divergence for or not

Returns : a list of triplets, for building a coo matrix
"""
def OperatorMatrix(diff,omega=None,mult=None, \
        h=1,bc=('Periodic',), divergenceForm=False,
        intrinsicDrift=False):
    
    # ----- Get the domain shape -----
    bounds = diff.shape[2:]
    dim = len(bounds)
    if len(bc)==1:
        bc = bc*dim
    elif len(bc)!=dim:
        raise ValueError("""OperatorMatrix error : 
        inconsistent boundary conditions""")
    
    if diff.shape[:2]!=(dim,dim):
        raise ValueError("OperatorMatrix error : inconsistent matrix dimensions")
        
    # -------- Decompose the tensors --------
    coef,offset = Selling.DecompP(diff)
    nCoef = coef.shape[0]
    
    # ------ Check bounds or apply periodic boundary conditions -------
    grid = np.mgrid[tuple(slice(0,n) for n in bounds)]
    bGrid = np.broadcast_to(np.reshape(grid,(dim,1,)+bounds), offset.shape)

    neighPos = bGrid + offset
    neighNeg = bGrid - offset
    insidePos = np.ones(coef.shape,dtype=bool)
    insideNeg = np.ones(coef.shape,dtype=bool)
    
    for neigh_,inside_ in zip( (neighPos,neighNeg), (insidePos,insideNeg) ):
        for neigh,inside,cond,bound in zip(neigh_,inside_,bc, bounds):
            if cond=='Periodic':
                neigh[neigh>=bound] -= bound
                neigh[neigh<0] +=bound
            else:
                inside = (neigh>=0) and (neigh<bound)
    
            
    # ------- Get the neighbor indices --------
    # Cumulative product in reverse order, omitting last term, beginning with 1
    cum = tuple(list(reversed(np.cumprod(list(reversed(bounds+(1,)))))))[1:]
    bCum = np.broadcast_to( np.reshape(cum, (dim,)+(1,)*(dim+1)), offset.shape)
    
    index = (bGrid*bCum).sum(0)
    indexPos = (neighPos*bCum).sum(0)
    indexNeg = (neighNeg*bCum).sum(0)
    
    # ------- Get the coefficients for the first order term -----
    if not omega is None:
        if intrinsicDrift:
            eta=omega
        else:
            eta = LP.dot_AV(LP.inverseP(diff),omega)
            
        scalEta = LP.dotP_VV(offset, 
            np.broadcast_to(np.reshape(eta,(dim,1,)+bounds),offset.shape)) 
        coefOmega = coef*scalEta

    # ------- Create the triplets ------
    
    # Second order part
    # TODO Non periodic boundary conditions ...
    
    coef = coef.flatten()
    index = index.flatten()
    indexPos = indexPos.flatten()
    indexNeg = indexNeg.flatten()
    
    if divergenceForm:
        row = np.concatenate((index, indexPos, index, indexPos))
        col = np.concatenate((index, index, indexPos, indexPos))
        data = np.concatenate((coef/2, -coef/2, -coef/2, coef/2))
        
        row  = np.concatenate(( row, index, indexNeg, index, indexNeg))
        col  = np.concatenate(( col, index, index, indexNeg, indexNeg))
        data = np.concatenate((data, coef/2, -coef/2, -coef/2, coef/2))
        
    else:
        row = np.concatenate( (index, index,    index))
        col = np.concatenate( (index, indexPos, indexNeg))
        data = np.concatenate((2*coef, -coef, -coef))
        
    # First order part, using centered finite differences
    if not omega is None:       
        coefOmega = coefOmega.flatten()
        row = np.concatenate((row, index,    index))
        col = np.concatenate((col, indexPos, indexNeg))
        data= np.concatenate((data,coefOmega/2,-coefOmega/2))
    
    if not mult is None:
        # TODO Non periodic boundary conditions
        size=np.prod(bounds)
        row = np.concatenate((row, range(size)))
        col = np.concatenate((col, range(size)))
        data= np.concatenate((data,mult.flatten()))
        
    return (row,col,data)
    
    
    
    