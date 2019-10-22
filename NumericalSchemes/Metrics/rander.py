from .base import Base
from .. import LinearParallel as lp

class Rander(Base):

	def __init__(self,m,w):
		self.m=m
		self.w=w

	def norm(self,v):
		return np.sqrt(lp.dot_VAV(v,m,v))+lp.dot_VV(v,w)
	
