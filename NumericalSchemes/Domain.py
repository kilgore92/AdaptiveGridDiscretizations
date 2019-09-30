from NumericalSchemes import FiniteDifferences as fd
from NumericalSchemes import AutomaticDifferentiation as ad
from NumericalSchemes import LinearParallel as lp

import numpy as np
from functools import reduce

class Domain(object):
	"""
	This class represents a domain from which one can query 
	a level set function and some related methods.

	The base class represents the full space R^d.
	"""

	def contains(self,x,h=0.):
		"""
		Equivalent to distance(x) + h < 0
		"""
		if (h==0. or 
			(h>0 and self.level_is_distance_inside) or 
			(h<0 and self.level_is_distance_outside) ):
			return self.level(x)+h<0
		else:
			return self.distance(x)+h<0

	def level(self,x):
		"""
		Level set function, negative inside, positive outside.
		"""
		return self.distance(x)

	def distance(self,x):
		"""
		Signed distance function, negative inside, positive outside.
		"""
		return np.inf

	def freeway(self,x,v):
		"""
		Output : Least h>0 such that x+h*v intersects the boundary.
		"""
		return np.inf

	@property
	def level_is_distance_inside(self):	return True

	@property
	def level_is_distance_outside(self): return True

	@property
	def level_is_distance(self):
		return self.level_is_distance_inside and self.level_is_distance_outside

	@property
	def is_convex(self):	return True

	@property
	def is_coconvex(self):	return True		 

class Ball(Domain):
	"""
	This class represents a ball shaped domain
	"""

	def __init__(self,center,radius=1.):
		if not isinstance(center,np.ndarray): center=np.array(center)
		self.center=center
		self.radius=radius

	def _centered(self,x):
		_center = fd.as_field(self.center,x.shape[1:],conditional=False)
		return x-_center

	def distance(self,x):
		_x = self._centered(x)
		return ad.Optimization.norm(_x,ord=2,axis=0)-radius

	def freeway(self,x,v):
		_x = self._centered(x)

		# Solve |x+hv|^2=r, which is a quadratic equation a h^2 + 2 b h + c =0
		a = lp.dot(v,v)
		b = lp.dot_VV(_x,v)
		c = lp.dot_VV(_x,_x)-self.radius

		delta = b*b-a*c

		far = delta<0
		delta[far]=0.
		sdelta = np.sqrt(delta)

		result = (-b-sdelta)/a
		result[far] = np.inf
		neg = result<0
		result[neg] = (-b+sdelta)/a
		neg = result<0
		result[neg] = np.inf

		return result

	@property
	def is_coconvex(self):	return True		 


class Box(Domain):
	"""
	This class represents a box shaped domain.
	"""

	def __init__(self,sides):
		if not isinstance(sides,np.ndarray): sides=np.array(sides)
		self._sides = sides
		self._center = sides.sum(axis=1)/2.
		self._hlen = (sides[:,1]-sides[:,0])/2.

	@property
	def sides(self): return self._sides
	@property
	def center(self): return self._center
	@property
	def edgelengths(self): return 2.*self._hlen

	def _centered(self,x,signs=False):
		_center = fd.as_field(self._center,x.shape[1:],conditional=False)
		_hlen = fd.as_field(self._hlen,x.shape[1:],conditional=False)
		_xc = x-_center
		result = np.abs(_xc)-_hlen
		return (result,np.sign(_xc)) if signs else result


	def level(self,x):
		_x = self._centered(x)
		return np.max(_x,axis=0)

	def distance(self,x):
		_x = self._centered(x)
		result = ad.Optimization.norm(np.maximum(_x,0.),ord=2,axis=0)
		pos = result==0.
		result[pos]=np.max(_x[:,pos],axis=0)

	def freeway(self,x,v):
		_x,signs = self._centered(x,signs=True)
		_v = v*signs

		null = _v==0
		_v[null]=np.nan

		result = _x/_v
		result[null] = np.inf
		result[result<0] = np.inf

		return np.min(result,axis=0)

	@property
	def level_is_distance_outside(self): return False

	@property
	def is_coconvex(self):	return True	


class AbsoluteComplement(Domain):
	"""
	This class represents the complement, in the entire space R^d, of an existing domain.
	"""
	def __init__(self,dom):
		self.dom = dom

	def contains(self,x,h=0.):
		return npt self.dom.contains(x,h)
	def level(self,x):
		return -self.dom.level(x)
	def distance(self,x):
		return -self.dom.distance(x)
	def freeway(self,x,v):
		return self.dom.freeway(x,v)

	@property
	def level_is_distance_inside(self):	return self.dom.level_is_distance_outside

	@property
	def level_is_distance_outside(self): return self.dom.level_is_distance_inside

	@property
	def is_convex(self):	return self.dom.is_coconvex

	@property
	def is_coconvex(self):	return self.dom.is_convex


class Intersection(Domain):
	"""
	This class represents an intersection of several subdomains.
	smooth : Describes the local behavior at the intersections of the subdomain boundaries
	"""
	def __init__(self,doms,smooth=False):

		self.doms=doms
		self.smooth = smooth

	def contains(self,x,h=0.):
		_contains = [dom.contains(x,h) for dom in self.doms]
		return reduce(np.logical_and,_contains)

	def level(self,x):
		levels = [dom.level(x) for dom in self.doms]
		return reduce(np.maximum,levels)

	def distance(self,x):

		distances = [dom.distances(x) for dom in self.doms]
		dist = reduce(np.maximum,distances)

		# Remove results expected to be invalid
		if not smooth: dist[dist>0] = np.inf

		return dist

	def freeway(self,x,v):
		assert False # Invalid outside, to work
		freeways = [dom.freeway(x,v) for dom in self.doms]
		return reduce(np.minimum,freeways)

	@property
	def level_is_distance_inside(self):	
		return all((dom.level_is_distance_inside for dom in self.doms))

	@property
	def level_is_distance_outside(self): 
		return smooth and all((dom.level_is_distance_outside for dom in self.doms))

	@property
	def is_convex(self):
		return all((dom.is_convex for dom in self.doms))

	@property
	def is_coconvex(self):
		return smooth and all((dom.is_coconvex for dom in self.doms))
	
def Complement(dom1,dom2,smooth=False):
	"""
	Relative complement dom1 \\ dom2
	"""
	return Intersection((dom1,AbsoluteComplement(dom2)),smooth)

def Union(doms,smooth=False):
	"""
	Union of various domains.
	smooth : Describes the local behavior at the intersections of the subdomain boundaries
	"""
	return AbsoluteComplement(Intersection([AbsoluteComplement(dom) for dom in doms],smooth))

class Dirichlet(object):
	"""
	Implements Dirichlet boundary conditions.
	Replaces all NaN values with values obtained on the boundary along the given direction
	"""

	def __init__(self,dom,bc,grid=None):
		"""
		Domain, boundary conditions, default grid.
		"""
		self.dom = dom
		self.bc = bc
		self.grid = grid

	def _grid(u,grid=None):
		if grid=None: grid=self.grid
		dim = len(grid)
		assert dim==0 or u.shape[-dim:]==grid.shape[1:]
		return grid

	def DiffUpwind(self,u,offsets,h,grid=None):
		grid=self._grid(u,grid)
		du = fd.DiffUpwind(u,offsets,h)
		mask = np.isnan(du)

		_grid = grid[:,mask]
		_offsets = offsets[:,mask]
		_h = self.dom.freeway(_grid,_mask)
		_x = _grid+_h*_offsets
		_bc = self.bc(_x)

		_du = (_bc-u)/h

		du[mask]=_du
		return du

	def DiffCentered(self,u,offsets,h,grid=None):
		"""
		Upwind differences are used at the boundary, in the direction of the boundary.
		"""
		assert False
		grid=self._grid(u,grid)
		du = fd.DiffCentered(u,offsets,h)
		mask = np.isnan(du)

	def Diff2(self,u,offsets,h,grid=None):
		"""
		Only first order at the boundary.
		"""
		assert False


		



#class Polygon2(Domain):
	"""
	Two dimensional polygon
	"""
"""
	def __init__(self,pts,convex=None):
		self.pts = pts
		if convex is None:
			assert False
		self.convex=convex

		assert False
		self.normals=None
		self.shifts=None

	def level(self,x):
		normals,shifts = (fd.as_field(e,x.shape[1:],conditional=False) 
			for e in (self.normals,self.shifts))

		return np.maximum(lp.dot_VA(x,normals)+shifts)

	def distance(self,x):
"""

