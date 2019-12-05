import numpy as np
from ... import AutomaticDifferentiation as ad
from ... import LinearParallel as lp
from ..base import Base


class ImplicitBase(object):
	"""
	Base class for a metric defined implicitly, 
	in terms of a level set function for the unit ball
	of the dual metric, and of a linear transformation.
	"""

	def __init__(self):
		self.a = None 

	def norm(self,v):
		return lp.dot_VV(v,self.gradient(v))

	def gradient(self,v):
		if self.a is None:
			return self._gradient(v)
		else:
			v,a = fd.common_field((v,a),(1,2))
			return lp.dot_AV(lp.tranpose(a),self._gradient(lp.dot_AV(a,v)))

	def inv_transform(self,a):
		if self.a is None:
			self.a = a
		else:
			self.a = lp.dot_AA(self.a,a)


	def _gradient(self,v):
		"""
		Gradient, ignoring self.a
		"""
		return ad.sequential_quadratic(v,self._dual_level,self._dual_params)


	def _dual_level(self,v,params=None):
		"""
		A level set function for the dual unit ball, ignoring self.a.
		Some parameters of the instance can be passed in argument, for AD purposes.
		"""
		raise ValueError('_dual_level is not implemented for this class')
		
	def _dual_params():
		return None
