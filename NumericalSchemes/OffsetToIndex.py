import numpy as np

# An array containing the coordinates of each point
def IndexArrayP(shape):
    

"""
Indices correspondence
- First index : Offset index, for a given point. We are assuming that the same number of offsets is requested for each point.
- Next d indices: position in the grid
- Last index : components of the index
"""

def OffsetToIndexP(offsets,bc):
    d = offsets.shape[-1]
    if offsets.ndim != d+2:
        raise ValueError("OffsetToIndexP error : inconsistent dimensions")
    
    n = offsets.shape[0] # Number of offsets for each point
    bounds = offsets.shape[1:-1]
    
    indices = np.zeros((n,)+bounds)
    for i in range(n):
        ind = indices[i]
        grid = np.mgrid(bounds)
        
        ind = points[0]
        for j in range(1,d):
            ind = ind*bounds[j]+points[j]
    
    
    