from .base import Base
from .riemann import Riemann
from .rander import Rander
from .. import LinearParallel as lp

class AsymQuad(Base):

	def __init__(self,m,w):
		self.m=m
		self.w=w

	def norm(self,v):
		v,m,w = misc.common_field((v,self.m,w),(1,2,1))
		return np.sqrt(lp.dot_VAV(v,m,v) + np.max(lp.dot_VV(m,w),0.)**2)

	def is_definite(self):
		return Riemann(self.m).is_definite()

	def dual(self):
		M = lp.inverse(self.m+lp.outer_self(w))
		wInv = lp.solve_AV(m,w)
		W = -wInv/np.sqrt(1.+lp.dot_VV(w,wInv))
		return AsymQuad(M,W)

	@property
	def ndim(self): return len(self.m)

	def inv_transform(self,a):
		return AsymQuad(Riemann(self.m).inv_transform(a),lp.dot_VA(w,a))

	def flatten(self):
		return Rander(self.m,self.w).flatten()

	@classmethod
	def expand(cls,arr):
		rd = Rander.expand(arr)
		return cls(rd.m,rd.w)