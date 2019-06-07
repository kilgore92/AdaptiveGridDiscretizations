# TODO : VTI,

# Hooke tensors are represented (as dxdxdxd tensors, but also) in Voigt notation, 
# but without the weird normalization by sqrt(2) and 2

import numpy as np
import itertools

def ExtractXZ(h):
	"""
	Extract a two dimensional Hooke tensor from a three dimensional one, 
	corresponding to a slice through the X and Z axes.
	"""
	return np.array([ 
		[h[1,1], h[1,3], h[1,5] ],
		[h[3,1], h[3,3], h[3,5] ],
		[h[5,1], h[5,3], h[5,5] ]
		])


def VTI(Vp,Vs,eps,delta): 
	"""
	X,Z slice of a Vertical Transverse Isotropic medium
	based on Thomsen parameters
	"""
	c33=Vp**2
	c44=Vs**2
	c11=c33*(1+2*eps)
	c13=-c44+np.sqrt( (c33-c44)**2+2*delta*c33*(c33-c44) )
	zero = 0.*Vs
	return np.array( [ [c11,c13,zero], [c13,c33,zero], [zero,zero,c44] ] )

Voigt2 = np.array([[0,2],[2,1]])
Voigt2i = np.array([[0,0],[1,1],[0,1]])

def Riemann(m):
	"""
	Rank deficient Hooke tensor,
	equivalent, for pressure waves, to the Riemannian metric defined by m squared.
	Shear waves are infinitely slow.
	"""
	a,b,c=m[0,0],m[1,1],m[0,1]
	return np.array( [ [a*a, a*b,a*c], [a*b, b*b, b*c], [a*c, b*c, c*c] ] )

def Rotate(h,t):
	r = np.array( [[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]] )
	return np.sum(np.array([ [ [
		h[Voigt2[i,j],Voigt2[k,l]]*r[i,ii]*r[j,jj]*r[k,kk]*r[l,ll]
		for (i,j,k,l) in itertools.product(range(2),repeat=4)]
		for (ii,jj) in Voigt2i] 
		for (kk,ll) in Voigt2i]
		), axis=2)

