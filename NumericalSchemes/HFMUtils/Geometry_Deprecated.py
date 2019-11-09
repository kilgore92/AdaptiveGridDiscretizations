
import numpy as np
import numbers

# ---------------- Array ordering for import/export ----------

# Maybe this would be better located in the c++ code, IO interface. 
# Because here, we cannot deal with fields depending only on some of the coordinates.
# Left as utility, but already deprecated.
def Coordinates_Components_Transpose(hfm,move_coordinates_first):
    hfm_old = {}
    dims = tuple(hfm['dims'].astype(int))
    dim=len(dims)
    if hfm['arrayOrdering']=='YXZ_RowMajor':
        if dim==2: dims = dims[1],dims[0]
        if dim>2: dims = dims[1],dims[0],dims[2:]
        
    for key,val in hfm.items():
        if not isinstance(val,np.ndarray): continue
        vdims= val.shape
        vdim = len(vdims)
        if vdim<=dim: continue
        if move_coordinates_first:
            if not vdims[vdim-dim:vdim] == dims: continue
            hfm_old[key]=val
            hfm[key] = np.transpose(val, tuple(range(vdim-dim,vdim)) + tuple(range(vdim-dim)) )
        else: # move_coordinates_last
            if not vdims[0:dim] == dims: continue
            hfm_old[key]=val
            hfm[key] = np.transpose( val, tuple(range(dim,vdim)) + tuple(range(dim)) )
    return hfm_old
