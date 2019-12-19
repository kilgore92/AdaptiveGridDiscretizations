import numpy as np
import itertools

from .. import misc
from ..riemann import Riemann
from ... import AutomaticDifferentiation as ad
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from ...FiniteDifferences import common_field
from .implicit_base import ImplicitBase



class Hooke(ImplicitBase):
	"""
	A norm defined by a Hooke tensor. 
	Often encountered in seismic traveltime tomography.
	"""
	def __init__(self,hooke,*args,**kwargs):
		super(Hooke,self).__init__(*args,**kwargs)
		self.hooke = hooke
		self._to_common_field()

	def is_definite(self):
		return Riemann(self.hooke).is_definite()

	@staticmethod
	def _vdim(hdim):
		"""Vector dimension from hooke tensor dimension"""
		vdim = int(np.sqrt(2*hdim))
		if (vdim*(vdim+1))//2!=hdim:
			raise ValueError("Incorrect hooke tensor")
		return vdim

	@property
	def vdim(self):
		return self._vdim(len(self.hooke))

	@property
	def shape(self): return self.hooke.shape[2:]
	
	def model_HFM(self):
		d = self.vdim
		suffix = "" if self.inverse_transformation is None else "Topographic"
		return f"Seismic{suffix}{d}"

	def flatten(self):
		hooke = misc.flatten_symmetric_matrix(self.hooke)
		if self.inverse_transformation is None: 
			return hooke
		else: 
			inv_trans= self.inverse_transformation.reshape((self.vdim**2,)+self.shape)
			return ad.concatenate((hooke,inv_trans),axis=0)

	@classmethod
	def expand(cls,arr):
		return cls(misc.expand_symmetric_matrix(arr))

	def __iter__(self):
		yield self.hooke
		for x in super(Hooke,self).__iter__():
			yield x

	def _to_common_field(self,*args,**kwargs):
		self.hooke,self.inverse_transformation = fd.common_field(
			(self.hooke,self.inverse_transformation),(2,2),*args,**kwargs)

	def _dual_params(self,*args,**kwargs):
		return fd.common_field((self.hooke,),(2,),*args,**kwargs)

	def _dual_level(self,v,params=None,relax=0.):
		if params is None: params = self._dual_params(v.shape[1:])

		# Contract the hooke tensor and covector
		hooke, = params
		Voigt,Voigti = self._Voigt,self._Voigti
		d = self.vdim
		m = ad.array([[
			sum(ad.toarray(v[j]*v[l]) * hooke[Voigt[i,j],Voigt[k,l]]
				for j in range(d) for l in range(d))
			for i in range(d)] for k in range(d)])

		# Evaluate det
		s = np.exp(-relax)
		ident = fd.as_field(np.eye(d),m.shape[2:],conditional=False)
		return ad.toarray(1.-s) - lp.det(ident - m*s) 

	def extract_xz(self):
		"""
		Extract a two dimensional Hooke tensor from a three dimensional one, 
		corresponding to a slice through the X and Z axes.
		"""
		assert(self.vdim==3)
		h=self.hooke
		return Hooke(np.array([ 
			[h[0,0], h[0,2], h[0,4] ],
			[h[2,0], h[2,2], h[2,4] ],
			[h[4,0], h[4,2], h[4,4] ]
			]))

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
	def from_Ellipse(cls,m):
		"""
		Rank deficient Hooke tensor,
		equivalent, for pressure waves, to the Riemannian metric defined by m ** -2.
		Shear waves are infinitely slow.
		"""
		assert(len(m)==2)
		a,b,c=m[0,0],m[1,1],m[0,1]
		return Hooke(np.array( [ [a*a, a*b,a*c], [a*b, b*b, b*c], [a*c, b*c, c*c] ] ))

	@classmethod
	def from_cast(cls,metric): 
		if isinstance(metric,cls):	return metric
		riemann = Riemann.from_cast(metric)
		
		m = riemann.dual().m
		assert not ad.is_ad(m)
		from scipy.linalg import sqrtm
		return cls.from_Ellipse(sqrtm(m))

	def _iter_implicit(self):
		yield self.hooke

	@property	
	def _Voigt(self):
		"""Direct Voigt indices"""
		if self.vdim==2:   return np.array([[0,2],[2,1]])
		elif self.vdim==3: return np.array([[0,5,4],[5,1,3],[4,3,2]])
		else: raise ValueError("Unsupported dimension")
	@property
	def _Voigti(self):
		"""Inverse Voigt indices"""
		if self.vdim==2:   return np.array([[0,0],[1,1],[0,1]])
		elif self.vdim==3: return np.array([[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]])
		else: raise ValueError("Unsupported dimension")


	def rotate(self,r):
		hooke,r = common_field((self.hooke,r),(2,2))
		Voigt,Voigti = self._Voigt,self._Voigti
		self.hooke = ad.array([ [ [
			hooke[Voigt[i,j],Voigt[k,l]]*r[ii,i]*r[jj,j]*r[kk,k]*r[ll,l]
			for (i,j,k,l) in itertools.product(range(self.vdim),repeat=4)]
			for (ii,jj) in Voigti] 
			for (kk,ll) in Voigti]
			).sum(axis=2)

	@staticmethod
	def _Mandel_factors(vdim,shape=tuple()):
		def f(k):	return 1. if k<vdim else np.sqrt(2.)
		hdim = (vdim*(vdim+1))//2
		factors = np.array([[f(i)*f(j) for i in range(hdim)] for j in range(hdim)])
		return fd.as_field(factors,shape,conditional=False)
	def to_Mandel(self):
		"""Introduces the sqrt(2) and 2 factors involved in Mandel's notation"""
		return self.hooke*self._Mandel_factors(self.vdim,self.shape)
	@classmethod
	def from_Mandel(cls,mandel):
		"""Removes the sqrt(2) and 2 factors involved in Mandel's notation"""
		return Hooke(mandel/cls._Mandel_factors(cls._vdim(len(mandel)),mandel.shape[2:]))

	@classmethod
	def from_orthorombic(cls,a,b,c,d,e,f,g,h,i):
		z=0.*a
		return cls(np.array([
		[a,b,c,z,z,z],
		[b,d,e,z,z,z],
		[c,e,f,z,z,z],
		[z,z,z,g,z,z],
		[z,z,z,z,h,z],
		[z,z,z,z,z,i]
		]))

	@classmethod
	def from_tetragonal(cls,a,b,c,d,e,f):
		return cls.from_orthorombic(a,b,c,a,c,d,e,e,f)

	@classmethod
	def from_hexagonal(cls,a,b,c,d,e):
		return cls.from_tetragonal(a,b,c,d,e,(a-b)/2)

# Densities in gram per cubic centimeter

	@classmethod
	def mica(cls,density=False):
		metric = cls.from_hexagonal(178.,42.4,14.5,54.9,12.2)
		rho = 2.79
		return (metric,rho) if density else metric

	@classmethod
	def stishovite(cls,density=False):
		metric = cls.from_tetragonal(453,211,203,776,252,302)
		rho = 4.29
		return (metric,rho) if density else metric

	@classmethod
	def olivine(cls,density=False):
		metric = cls.from_orthorombic(323.7,66.4,71.6,197.6,75.6,235.1,64.6,78.7,79.0)
		olivine_rho = 3.311
		return (metric,rho) if density else metric

	@classmethod
	def from_Reduced(cls,metric):
		"""Generate full Hooke tensor from reduced algebraic form.
		Warning : Reduced to Hooke conversion may induce approximations."""
		from .reduced import Reduced
		l,q,c = metric.linear,metric.quadratic,metric.cubic
		z = np.zeros(l.shape[1:])
		if metric.vdim==2:
			hooke = ad.array([ 
				[l[0],z],
				[z,l[1]]
				])
			raise ValueError("TODO : correct implementation") #Note : hooke shape should be 3x3
		elif metric.vdim==3:
			hooke = ad.array([
				[l[0],z,z],
				[z,l[1],z],
				[z,z,l[2]]
				])



			raise ValueError("TODO : correct implementation") #Note : hooke shape should be 6x6
		else:
			raise ValueError("Unsupported dimension")
		return cls(hooke,*super(Reduced,metric).__iter__())











