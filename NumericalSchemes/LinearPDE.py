import numpy as np
import scipy.sparse as sp

from NumericalSchemes import Selling
from NumericalSchemes import LinearParallel as LP

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
    coef,offset = Selling.Decomposition(diff)
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
                inside = np.logical_and(neigh>=0, neigh<bound)
    
    print(insidePos)

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
            eta = LP.dot_AV(LP.inverse(diff),omega)
            
        scalEta = LP.dot_VV(offset.astype(float), 
            np.broadcast_to(np.reshape(eta,(dim,1,)+bounds),offset.shape)) 
        coefOmega = coef*scalEta

    # ------- Create the triplets ------
    
    # Second order part
    # Nemann : remove all differences which are not inside (a.k.a multiply coef by inside)
    # TODO : Dirichlet : set to zero the coef only for the outside part
    
    coef = coef.flatten()
    index = index.flatten()
    indexPos = indexPos.flatten()
    indexNeg = indexNeg.flatten()
    iP, iN = insidePos.flatten().astype(float), insideNeg.flatten().astype(float)

    if divergenceForm:
        row = np.concatenate((index, indexPos, index, indexPos))
        col = np.concatenate((index, index, indexPos, indexPos))
        data = np.concatenate((iP*coef/2, -iP*coef/2, -iP*coef/2, iP*coef/2))
        
        row  = np.concatenate(( row, index, indexNeg, index, indexNeg))
        col  = np.concatenate(( col, index, index, indexNeg, indexNeg))
        data = np.concatenate((data, iN*coef/2, -iN*coef/2, -iN*coef/2, iN*coef/2))
        
    else:
        row = np.concatenate( (index, index,    index))
        col = np.concatenate( (index, indexPos, indexNeg))
        data = np.concatenate((iP*coef+iN*coef, -iP*coef, -iN*coef))
    

    # First order part, using centered finite differences
    if not omega is None:       
        coefOmega = coefOmega.flatten()
        row = np.concatenate((row, index,    index))
        col = np.concatenate((col, indexPos, indexNeg))
        data= np.concatenate((data,iP*coefOmega/2,-iN*coefOmega/2))
    
    if not mult is None:
        # TODO Non periodic boundary conditions
        size=np.prod(bounds)
        row = np.concatenate((row, range(size)))
        col = np.concatenate((col, range(size)))
        data= np.concatenate((data,mult.flatten()))

    nz = data!=0
        
    return (row[nz],col[nz],data[nz])
    
    
    
    