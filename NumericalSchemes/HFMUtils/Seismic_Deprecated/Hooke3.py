import numpy as np
import scipy.linalg
import itertools

# Hooke tensors are represented in Voigt notation, 
# but without the weird normalization by sqrt(2) and 2

Voigt3 = np.array([[0,5,4],[5,1,3],[4,3,2]])
Voigt3i = np.array([[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]])

def Riemann(m):
	"""
	Rank deficient Hooke tensor,
	equivalent, for pressure waves, to the Riemanian metric defined by m squared.
	Shear waves are infinitely slow.
	"""
	a = [m[i,j] for (i,j) in Voigt3i]
	return np.array([ [a[i]*a[j] for i in range(6)] for j in range(6)])


def Rotate(h,r):
	"""
	Rotate a three dimensional hooke tensor
	"""
	return np.sum(np.array([ [ [
		h[Voigt3[i,j],Voigt3[k,l]]*r[i,ii]*r[j,jj]*r[k,kk]*r[l,ll]
		for (i,j,k,l) in itertools.product(range(3),repeat=4)]
		for (ii,jj) in Voigt3i] 
		for (kk,ll) in Voigt3i]
		), axis=2)


def RotationMatrix(axis, theta):
	"""
	Three dimensional rotation matrix, with given axis and angle.
	Adapted from https://stackoverflow.com/a/6802723
	"""
	axis = np.asarray(axis)
	axis = axis / np.linalg.norm(axis,axis=0)
	a = np.cos(theta / 2.0)
	b, c, d = -axis * np.sin(theta / 2.0)
	aa, bb, cc, dd = a * a, b * b, c * c, d * d
	bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
	return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
	                 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
	                 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
#	return scipy.linalg.expm(np.cross(np.eye(3), axis/scipy.linalg.norm(axis)*theta)) # Alternative


# Some classical media

def Orthorombic(a,b,c,d,e,f,g,h,i):
	z=0.*a
	return np.array([
		[a,b,c,z,z,z],
		[b,d,e,z,z,z],
		[c,e,f,z,z,z],
		[z,z,z,g,z,z],
		[z,z,z,z,h,z],
		[z,z,z,z,z,i]
		])

def Tetragonal(a,b,c,d,e,f):
	return Orthorombic(a,b,c,a,c,d,e,e,f)

def Hexagonal(a,b,c,d,e):
	return Tetragonal(a,b,c,d,e,(a-b)/2)

Mica = Hexagonal(178.,42.4,14.5,54.9,12.2)
Stishovite = Tetragonal(453,211,203,776,252,302)
Olivine = Orthorombic(323.7,66.4,71.6,197.6,75.6,235.1,64.6,78.7,79.0)

# Densities in gram per cubic centimeter
Mica_rho = 2.79
Stishovite_rho = 4.29
Olivine_rho = 3.311
