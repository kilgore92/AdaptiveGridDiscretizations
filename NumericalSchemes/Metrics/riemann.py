from .base import Base
from .. import LinearParallel as lp

class Riemann(Base):

	def __init__(self,m):
		self.m=m

	def norm(self,v):
		return np.sqrt(lp.dot_VAV(v,m,v))

	def dual(self):
		return Riemann(lp.inverse(self.m))