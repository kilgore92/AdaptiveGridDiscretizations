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
	
	def dual(self):
		raise ValueError("dual is not implemented for this norm")

	@property
	def ndim(self):
		raise ValueError("ndim is not implemented for this norm")

# ---- Well posedness related methods -----
	def is_definite(self):
		"""
		Wether norm(u)=0 implies u=0. 
		"""
		raise ValueError("is_definite is not implemented for this norm")

	def anisotropy(self):
		"""
		Sharp upper bound on norm(u)/norm(v), 
		for any unit vectors u and v.
		"""
		raise ValueError("anisotropy is not implemented for this norm")

	def anisotropy_bound(self):
		"""
		Upper bound on norm(u)/norm(v), 
		for any unit vectors u and v.
		"""
		return self.anisotropy()
# ---- Causality and acuteness related methods ----

	def cos_asym(self,u,v):
		"""
		Generalized cosine defined by the metric, 
		asymmetric variant defined as 
		<grad F(u), v> / F(v)
		"""
		u,v=(ad.toarray(e) for e in (u,v))
		return lp.dot_VV(self.gradient(u),v)/self.norm(v)

	def cos(self,u,v):
		"""
		Generalized cosine defined by the metric.
		"""
		u,v=(ad.toarray(e) for e in (u,v))
		gu,gv=self.gradient(u),self.gradient(v)
		guu,guv = lp.dot_VV(gu,u),lp.dot_VV(gu,v)
		gvu,gvv = lp.dot_VV(gv,u),lp.dot_VV(gv,v)
		return np.minimum(guv/gvv,gvu/guu)

	def angle(self,u,v):
		c = ad.toarray(self.cos(u,v))
		mask=c < -1.
		c[mask]=0.
		result = ad.toarray(np.arccos(c))
		result[mask]=np.inf
		return result

# ---- Geometric transformations ----

	def inv_transform(self,a):
		"""
		Affine transformation of the norm. 
		The new unit ball is the inverse image of the previous one.
		"""
		raise ValueError("Affine transformation not implemented for this norm")

	def transform(self,a):
		"""
		Affine transformation of the norm.
		The new unit ball is the direct image of the previous one.
		"""
		return self.inv_transform(lp.inverse(a))

	def rotate(self,r):
		"""
		Rotation of the norm, by a given rotation matrix.
		The new unit ball is the direct image of the previous one.
		"""
		return self.transform(r)

	def rotate_by(self,theta,axis=None):
		"""
		Rotation of the norm.
		Dimension 2 : by a given angle.
		Dimension 3 : by a given angle, along a given axis.
		Three dimensional rotation matrix, with given axis and angle.
		Adapted from https://stackoverflow.com/a/6802723
		"""
		if axis is None:
			c,s=np.cos(theta),np.sin(theta)
			return self.rotate(np.array([[c,-s],[s,c]]))
		else:
			axis = np.asarray(axis)
			axis = axis / np.linalg.norm(axis,axis=0)
			a = np.cos(theta / 2.0)
			b, c, d = -axis * np.sin(theta / 2.0)
			aa, bb, cc, dd = a * a, b * b, c * c, d * d
			bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
			return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
	                 		[2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
	                 		[2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
#	return scipy.linalg.expm(np.cross(np.eye(3), axis/scipy.linalg.norm(axis)*theta)) # Alternative


# ---- Import and export ----

	def flatten(self):
		raise ValueError("Flattening not implemented for this norm")

	@classmethod
	def expand(cls,arr):
		raise ValueError("Expansion not implemented for this norm")

	def to_hfm(self):
		"""
Formats a metric for the HFM library. 
This may include flattening some symmetric matrices, 
concatenating with vector fields, and moving the first axis last.
"""
		return np.moveaxis(self.flatten(),0,-1)

	@classmethod
	def from_hfm(cls,arr):
		return cls.expand(np.moveaxis(arr,-1,0))

""" 
Possible additions : 
 - shoot geodesic (with a grid), 
"""