import numpy as np
from .SparseAD import spAD

def TakeWithOffset(u,offset,padding=np.inf,uniform=None,autodiff=True):
	"""
	Returns the value of u at current position + offset. Also returns the coefficients and indices.
	Set padding=None for periodic boundary conditions
	"""
	assert(offset.shape[0]==u.ndim)
	if uniform is None:
		uniform = not ((offset.ndim >= 1+u.ndim) and (offset.shape[-u.ndim:]==u.ndim))

	grid = np.mgrid[tuple(slice(n) for n in u.shape)]
	grid = grid.reshape( (u.ndim,) + (1,)*(offset.ndim-1-u.ndim*int(not uniform))+u.shape)

	neigh = grid + (offset.reshape(offset.shape + (1,)*u.ndim) if uniform else offset)
	inside = np.full(neigh.shape[1:],True) # Weither neigh falls out the domain

	if padding is None: # Apply periodic bc
		for coord,bound in zip(neigh,u.shape):
			coord %= bound
	else: #identify bad indices
		for coord,bound in zip(neigh,u.shape):
			inside =np.logical_and(inside, np.logical_and(coord>=0,coord<bound))
#			inside = np.logical_and.reduce(inside, coord>=0, coord<bound)

	neighIndex = np.ravel_multi_index(neigh, u.shape, mode = 'clip')
	result = np.full(inside.shape,padding if padding is not None else u.flatten()[0])
	result[inside] = u.flatten()[neighIndex[inside]]

	return spAD(result,inside,neighIndex) if autodiff else result

def AlignedSum(u,offset,multiples,weights,**kwargs):
	"""Returns sum along the direction offset, with specified multiples and weights"""
	return sum(TakeWithOffset(u,mult*offset,**kwargs)*weight for mult,weight in zip(multiples,weights))

def Diff2(u,offset,gridScale=1.,**kwargs):
	"""Second order finite difference in the specidied direction"""
	return AlignedSum(u,offset,(1,0,-1),np.array((1,-2,1))/gridScale**2,**kwargs)

def DiffCentered(u,offset,gridScale=1.,**kwargs):
	"""Centered first order finite difference in the specified direction"""
	return AlignedSum(u,offset,(1,-1),np.array((1,-1))/(2*gridScale),**kwargs)

def DiffUpwind(u,offset,gridScale=1.,**kwargs):
	"""Upwind first order finite difference in the specified direction"""
	return AlignedSum(u,offset,(1,0),np.array((1,-1))/gridScale,**kwargs)
