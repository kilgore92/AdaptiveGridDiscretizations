import numpy as np
from .base import Base
from . import misc
from .riemann import Riemann
from .. import LinearParallel as lp

class Hooke(Base):
	"""
A norm defined by a Hooke tensor. 
Often encountered in seismic traveltime tomography.
"""
	def __init__(self,hooke):
		self.hooke = hooke 

	def is_definite(self):
		return Riemann(self.hooke).is_definite()

	@property
	def ndim(self):
		if len(self.hooke)==3: return 2
		elif len(self.hooke)==6: return 3
		else: raise ValueError("Incorrect hooke tensor")

	def flatten(self):
		return misc.flatten_symmetric_matrix(self.hooke)

	@classmethod
	def expand(cls,arr):
		return cls(misc.expand_symmetric_matrix(arr))

	def extract_xz_3(self):
		"""
	Extract a two dimensional Hooke tensor from a three dimensional one, 
	corresponding to a slice through the X and Z axes.
	"""
		assert(self.ndim==3)
		h=self.hooke
		return np.array([ 
			[h[0,0], h[0,2], h[0,4] ],
			[h[2,0], h[2,2], h[2,4] ],
			[h[4,0], h[4,2], h[4,4] ]
			])

	@classmethod
	def from_VTI_2(cls,Vp,Vs,eps,delta):
		"""
	X,Z slice of a Vertical Transverse Isotropic medium
	based on Thomsen parameters
	"""
		c33=Vp**2
		c44=Vs**2
		c11=c33*(1+2*eps)
		c13=-c44+np.sqrt( (c33-c44)**2+2*delta*c33*(c33-c44) )
		zero = 0.*Vs
		return cls(np.array( [ [c11,c13,zero], [c13,c33,zero], [zero,zero,c44] ] ))

	@classmethod
	def from_Riemann(m):
		"""
	Rank deficient Hooke tensor,
	equivalent, for pressure waves, to the Riemannian metric defined by m squared.
	Shear waves are infinitely slow.
	"""
		assert(len(m)==2)
		a,b,c=m[0,0],m[1,1],m[0,1]
		return np.array( [ [a*a, a*b,a*c], [a*b, b*b, b*c], [a*c, b*c, c*c] ] )



	def rotate(self,r):
		Voigt2 = np.array([[0,2],[2,1]])
		Voigt2i = np.array([[0,0],[1,1],[0,1]])

		Voigt3 = np.array([[0,5,4],[5,1,3],[4,3,2]])
		Voigt3i = np.array([[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]])

		Voigt,Voigti = (Voigt2,Voigt2i) if self.ndim==2 else (Voigt3,Voigt3i)

		return np.sum(np.array([ [ [
			h[Voigt[i,j],Voigt[k,l]]*r[i,ii]*r[j,jj]*r[k,kk]*r[l,ll]
			for (i,j,k,l) in itertools.product(range(2),repeat=4)]
			for (ii,jj) in Voigti] 
			for (kk,ll) in Voigti]
			), axis=2)


	@classmethod
	def from_orthorombic_3(cls,a,b,c,d,e,f,g,h,i):
		z=0.*a
		return np.array([
		[a,b,c,z,z,z],
		[b,d,e,z,z,z],
		[c,e,f,z,z,z],
		[z,z,z,g,z,z],
		[z,z,z,z,h,z],
		[z,z,z,z,z,i]
		])

	@classmethod
	def from_tetragonal_3(cls,a,b,c,d,e,f):
		return cls.from_orthorombic_3(a,b,c,a,c,d,e,e,f)

	@classmethod
	def from_hexagonal_3(cls,a,b,c,d,e):
		return cls.from_tetragonal_3(a,b,c,d,e,(a-b)/2)

# Densities in gram per cubic centimeter

	@classmethod
	@property
	def mica_3(cls):
		mica_rho = 2.79
		return cls.from_hexagonal_3(178.,42.4,14.5,54.9,12.2),mica_rho

	@classmethod
	@property
	def stishovite_3(cls):
		stishovite_rho = 4.29
		return cls.from_tetragonal_3(453,211,203,776,252,302),stishovite_rho

	@classmethod
	@property
	def olivine_3(cls):
		olivine_rho = 3.311
		return cls.from_orthorombic_3(323.7,66.4,71.6,197.6,75.6,235.1,64.6,78.7,79.0),olivine_rho