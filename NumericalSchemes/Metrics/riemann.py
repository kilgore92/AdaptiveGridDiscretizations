import numpy as np

from . import misc
from .base import Base
from .. import LinearParallel as lp

class Riemann(Base):

	def __init__(self,m):
		self.m=m

	def norm(self,v):
		v,m,w = misc.common_field((v,self.m),(1,2))
		return np.sqrt(lp.dot_VAV(v,m,v))

	def is_definite(self):
		return np.linalg.eigvals(np.moveaxis(self.m,(0,1),(-2,-1))).min(axis=-1)>0

	def dual(self):
		return Riemann(lp.inverse(self.m))

	@property
	def ndim(self): return len(self.m)

	def inv_transform(self,a):
		return Riemann(lp.dot_AA(lp.transpose(a),lp.dot_AA(m,a)))

	def flatten(self):
		return misc.flatten_symmetric_matrix(self.m)

	@classmethod
	def expand(cls,arr):
		m = misc.expand_symmetric_matrix(arr)
		d=len(m)
		assert(len(arr)==(d*(d+1))//2)
		return cls(m)


