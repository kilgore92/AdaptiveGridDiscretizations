import numpy as np
#from . import SparseAutomaticDifferentiation as spAD

def OffsetToIndex(shape,offset, mode='clip', uniform=None):
	"""
	Returns the value of u at current position + offset. Also returns the coefficients and indices.
	Set padding=None for periodic boundary conditions
	"""
	ndim = len(shape)
	assert(offset.shape[0]==ndim)
	if uniform is None:
		uniform = not ((offset.ndim > ndim) and (offset.shape[-ndim:]==shape))

	grid = np.mgrid[tuple(slice(n) for n in shape)]
	grid = grid.reshape( (ndim,) + (1,)*(offset.ndim-1-ndim*int(not uniform))+shape)

	neigh = grid + (offset.reshape(offset.shape + (1,)*ndim) if uniform else offset)
	inside = np.full(neigh.shape[1:],True) # Weither neigh falls out the domain

	if mode=='wrap': # Apply periodic bc
		for coord,bound in zip(neigh,shape):
			coord %= bound
	else: #identify bad indices
		for coord,bound in zip(neigh,shape):
			inside = np.logical_and.reduce( (inside, coord>=0, coord<bound) )

	neighIndex = np.ravel_multi_index(neigh, shape, mode=mode)
	return neighIndex, inside

def TakeAtOffset(u,offset, padding=0., **kwargs):
	mode = 'wrap' if padding is None else 'clip'
	neighIndex, inside = OffsetToIndex(u.shape,offset,mode=mode, **kwargs)

	values = u.flatten()[neighIndex]
	if padding is not None:
		values[np.logical_not(inside)] = padding
#		values = spAD.replace_at(values,np.logical_not(inside),padding)
	return values

def AlignedSum(u,offset,multiples,weights,**kwargs):
	"""Returns sum along the direction offset, with specified multiples and weights"""
	return sum(TakeAtOffset(u,mult*np.array(offset),**kwargs)*weight for mult,weight in zip(multiples,weights))

def Diff2(u,offset,gridScale=1.,**kwargs):
	"""Second order finite difference in the specidied direction"""
	return AlignedSum(u,offset,(1,0,-1),np.array((1,-2,1))/gridScale**2,**kwargs)

def DiffCentered(u,offset,gridScale=1.,**kwargs):
	"""Centered first order finite difference in the specified direction"""
	return AlignedSum(u,offset,(1,-1),np.array((1,-1))/(2*gridScale),**kwargs)

def DiffUpwind(u,offset,gridScale=1.,**kwargs):
	"""Upwind first order finite difference in the specified direction"""
	return AlignedSum(u,offset,(1,0),np.array((1,-1))/gridScale,**kwargs)
