import numpy as np
from .. import AutomaticDifferentiation as ad
from .. import LinearParallel as lp

class Base(object):
	"""
	Base class for a metric
	"""

	def norm(self,v):
		"""
		Norm defiend by the metric. 
		Expected to be 1-homogeneous w.r.t. v
		"""
		raise ValueError("""Error : norm must be specialized in subclass""")

	def gradient(self,v):
		"""
		Gradient of the norm defined by the metric
		"""
		v_ad = ad.Dense.identity(constant=v,shape_free=(len(v),))
		return np.moveaxis(self.norm(v_ad).coef,-1,0)

	def cos_asym(self,u,v):
		"""
		Generalized cosine defined by the metric, 
		asymmetric variant defined as 
		<grad F(u), v> / F(v)
		"""
		return lp.dot_VV(self.gradient(u),v)/self.norm(v)

	def cos(self,u,v):
		"""
		Generalized cosine defined by the metric.
		"""
		gu,gv=self.gradient(u),self.gradient(v)
		guu,guv = lp.dot_VV(gu,u),lp.dot_VV(gu,v)
		gvu,gvv = lp.dot_VV(gv,u),lp.dot_VV(gv,v)
		return np.minimum(guv/guu,gvu/guu)

""" Possible additions : 
 - shoot geodesic (with a grid), 
 - dual norm (with a Newton method). 
 - rotate (with a matrix)
 - rotate2 (with an angle)
 - to_hfm
 - from_hfm