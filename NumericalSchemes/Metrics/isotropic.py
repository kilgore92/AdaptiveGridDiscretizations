from .base import Base
from .. import AutomaticDifferentiation as ad

class Isotropic(Base):
	"""
	An isotropic metric, defined through a cost function.
	"""

	def __init__(self,cost):
		self.cost = cost

	@classmethod
	def from_speed(cls,speed):
		return cls(1./speed)

	def dual(self):
		return self.from_speed(self.cost)

	def norm(self,v):
		return self.cost*ad.Optimization.norm(v,ord=2,axis=0)

	def is_definite(self):
		return self.cost>0.

	@property
	def ndim(self): 
		if self.cost.ndim>0:
			return self.cost.ndim
		elif hasattr(self,'_ndim'):
			return self._ndim
		else:
			raise ValueError("Could not determine dimension of isotropic metric")

	@ndim.setter
	def ndim(self,ndim):
		if self.cost.ndim>0:
			assert(self.cost.ndim==ndim)
		else:
			self._ndim = ndim

	def rotate(self,a):
		return self

	def flatten(self):		return self.cost
	@classmethod
	def expand(cls,arr):	return cls(arr)

	def to_hfm(self):		return self.cost
	@classmethod
	def from_hfm(cls,arr):	return cls(arr)


	# TODO : upwind gradient from HFM AD info (with grid)