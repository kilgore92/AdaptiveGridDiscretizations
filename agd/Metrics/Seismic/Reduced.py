import numpy as np
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ..riemann import Riemann
from .implicit_base import ImplicitBase


class Reduced(ImplicitBase):
	"""
	A family of reduced models appearing in seismic tomography, 
	based on assuming that some of the coefficients of the Hooke tensor vanish.

	The dual ball is defined by an equation of the form 
	l(X^2,Y^2,Z^2) + q(X^2,Y^2,Z^2) + c X^2 Y^2 Z^2 = 1
	where l is linear, q is quadratic, and c is a cubic coefficient.
	X,Y,Z are the coefficients of the input vector, perhaps subject to a linear transformation.
	"""

	def __init__(self,linear,quadratic=None,cubic=None,*args,**kwargs):
		super(Reduced,self).__init__(*args,**kwargs)
		self.linear=ad.array(linear)
		self.quadratic=None if quadratic is None else ad.array(quadratic)
		self.cubic=None if cubic is None else ad.array(cubic)
		assert cubic is None or self.vdim==3
		self._to_common_field()

	@property
	def vdim(self):
		return len(self.linear)
	
	@property
	def shape(self):
		return self.linear.shape[1:]

	def _dual_level(self,v,params=None,relax=0.):
		if params is None: params = self._dual_params(v.shape[1:])
		s = np.exp(-relax)
		l,q,c = params
		v2 = v**2
		result = lp.dot_VV(l,v2) - 1
		if q is not None:
			result += s*lp.dot_VAV(v2,q,v2)
		if c is not None:
			assert self.vdim==3
			result += s**2*c*v2.prod()
		return result

	def _dual_params(self,*args,**kwargs):
		return fd.common_field((self.linear,self.quadratic,self.cubic),(1,2,0),*args,**kwargs)

	def __iter__(self):
		yield self.linear
		yield self.quadratic
		yield self.cubic
		for x in super(Reduced,self).__iter__(): 
			yield x

	def _to_common_field(self,*args,**kwargs):
		self.linear,self.quadratic,self.cubic,self.inv_transform = fd.common_field(
			(self.linear,self.quadratic,self.cubic,self.inv_transform),(1,2,0,2),*args,**kwargs)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		riemann = Riemann.from_cast(metric)
		assert not ad.is_ad(riemann.m)
		e,a = np.linalg.eig(riemann.m)
		result = Reduced(e)
		result.inv_transform(a)
		return result

	@classmethod
	def from_Hooke(cls,metric):
		"""
		Generate reduced algebraic form from full Hooke tensor.
		Warning : Hooke to Reduced conversion requires that some 
		coefficients of the Hooke tensor vanish, and may induce approximations.
		"""
		from .hooke import Hooke
		hooke = metric.hooke
		if metric.vdim==2:
			linear = ad.array([hooke[0,0],hooke[1,1]])
			quadratic = ad.array([[hooke[0,0],hooke[0,2]],[hooke[2,0],hooke[2,2]]])
			cubic = None
			raise ValueError("TODO : correct implementation")
		elif metric.vdim==3:
			linear = ad.array([hooke[0,0],hooke[1,1],hooke[2,2]])
			quadratic = None
			cubic = None
			if np.any((hooke[1,3]!=0.,hooke[2,4]!=0.)):
				raise ValueError("Impossible conversion")
			raise ValueError("TODO : correct implementation")
		else:
			raise ValueError("Unsupported dimension")

		return cls(linear,quadratic,cubic,*super(Hooke,metric).__iter__())






