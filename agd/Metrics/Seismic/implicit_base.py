import numpy as np
from ... import AutomaticDifferentiation as ad
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from ..base import Base


class ImplicitBase(Base):
	"""
	Base class for a metric defined implicitly, 
	in terms of a level set function for the unit ball
	of the dual metric, and of a linear transformation.
	"""

	def __init__(self):
		self.a = None 
		self.niter_sqp = 6
		self.relax_sqp = ()

	def norm(self,v):
		return lp.dot_VV(v,self.gradient(v))

	def gradient(self,v):
		if self.a is None:
			return self._gradient(v)
		else:
			v,a = fd.common_field((v,self.a),(1,2))
			return lp.dot_AV(lp.transpose(a),self._gradient(lp.dot_AV(a,v)))

	def inv_transform(self,a):
		if self.a is None:
			self.a = a
		else:
			self.a = lp.dot_AA(self.a,a)


	def _gradient(self,v):
		"""
		Gradient, ignoring self.a
		"""
		v=ad.array(v)
		return sequential_quadratic(v,self._dual_level,params=self._dual_params(),
			niter=self.niter_sqp,relax=self.relax_sqp)


	def _dual_level(self,v,params=None,relax=0):
		"""
		A level set function for the dual unit ball, ignoring self.a.
		Some parameters of the instance can be passed in argument, for AD purposes.
		Parameter s is for a relaxation of the level set. 0->exact, np.inf->easy (quadratic).
		"""
		raise ValueError('_dual_level is not implemented for this class')
		
	def _dual_params(self):
		"""
		The parameters to be passed to _dual_level.
		"""
		return None


def sequential_quadratic(v,f,niter,x=None,params=tuple(),relax=tuple()):
	"""
	Maximizes <x,v> subject to the constraint f(x,*params)<=0, 
	using sequential quadratic programming.
	x : initial guess.
	relax : relaxation parameters to be used in a preliminary path following phase.
	params : to be passed to evaluated function. Special treatment if ad types.
	"""
	if x is None: x=np.zeros(v.shape)

	params_noad = tuple(ad.remove_ad(val) for val in params) 

	x_ad = ad.Dense2.identity(constant=x,shape_free=(len(x),))

	# Fixed point iterations 
	def step(val,V,D,v):
		M = lp.inverse(D)
		k = np.sqrt((lp.dot_VAV(V,M,V)-2*val)/lp.dot_VAV(v,M,v))
		return lp.dot_AV(M,k*v-V)

	for r in relax + (0.,)*niter:
		f_ad = f(x_ad,params_noad,relax=r)
		x_ad += step(f_ad.value,f_ad.gradient(),f_ad.hessian(),v)

	x=x_ad.value

	# Terminal iteration to introduce ad information
	adtype = ad.is_ad(params,iterables=(tuple,))
	if adtype:
		shape_bound = x.shape[1:]
		params_dis = tuple(ad.disassociate(value,shape_bound=shape_bound) 
			if isinstance(value,np.ndarray) else value for value in params)
		x_ad = ad.Dense2.identity(constant=ad.disassociate(x,shape_bound=shape_bound))

		f_ad = f(x_ad,params_dis,1.)
		x = x + step(ad.associate(f_ad.value), ad.associate(f_ad.gradient()), ad.associate(f_ad.hessian()), v)

	return x

