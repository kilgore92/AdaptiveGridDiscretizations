import numpy as np 
from .base import Base
from .hooke import Hooke
from . import misc

class HookeTopo(Base):
	"""
A norm defined by a Hooke tensor, and the gradient of a height map, which accounts for topographic information.
"""
	def __init__(self,hooke,height_grad):
		self.hooke = hooke
		self.height_grad = np.array(height_grad)

	def flatten(self):
		return np.concatenate( (misc.flatten_symmetric_matrix(self.hooke),self.height_grad), axis=0)

	@classmethod
	def expand(cls,arr):
		hooke = misc.expand_symmetric_matrix(arr,extra_length=True)
		d=Hooke(hooke).ndim
		d_hooke = (d*(d+1))//2
		assert(len(arr)==(d_hooke*(d_hooke+1))//2 + d)
		height_grad = arr[-d:]
		return cls(hooke,height_grad)







