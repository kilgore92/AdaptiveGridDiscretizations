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

	# TODO : upwind gradient from HFM AD info (with grid)