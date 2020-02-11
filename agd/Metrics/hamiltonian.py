"""
This module implements some basic functionality for solving ODEs derived from a 
Hamiltonian in a manner compatible with automatic differentiation.
(Flow computation, symplectic schemes, etc)

Recall that Hamilton's equations read 
dq/dt =  dH/dp
dp/dt = -dH/dq

Given a metric F, the corresponding canonical Hamiltonian is defined as 
H(q,p) = (1/2) F^*_q(p)^2.
In written words, the Hamiltonian is the half square of the dual metric.
"""

import scipy.sparse

from .. import LinearParallel as lp
from .. import AutomaticDifferentiation as ad
from .base import Base

lo = ad.left_operand

def fixedpoint(f,x,tol=1e-8,nitermax=100):
	"""
	Iterates the function f on the data x until a fixed point is found, 
	up to prescribed tolerance, or the maximum number of iterations is reached.
	"""
	norm_infinity = ad.Optimization.norm_infinity
	x_old = x
	for i in range(nitermax):
		x=f(x)
		if norm_infinity(x-xold)<tol: break
		xold = x
	return x


class Hamiltonian(object):
	def __init__(self, H, shape_free=None):
		"""
		Inputs:
		- H : the hamiltonian, which may be either:
			* a metric
		 	* a callable function
			* a pair of callable functions, for a separable hamiltonian.
				(in that case, may also be scalars or matrices, for quadratic hamiltonians)
		- shape_free (optional) : the shape of the position and impulsion
		"""

#		 - Structure : the hamiltonian structure, w.r.t. the first and second variable.
#		   Possibilities include "scalar","matrix","sparse" and None
		
		is isinstance(H,list): H = tuple(H)
		self._H = H
		if self.is_separable: assert len(H)==2

		if self.is_metric:
			assert shape_free is None
			self.vdim = self._H.vdim
		else:
			self.shape_free = shape_free

#		assert isinstance(structure,tuple) and len(structure)==2
#		for e in structure: 
#			assert e in ["scalar","matrix","sparse",None]
#		self.structure = structure
#		self.shape_free = shape_free
#		if isinstance(H,Base):
#			if self.shape_free is None:
#				self.shape_free = (H.vdim,)
#			else:
#				assert self.shape_free==(H.vdim,)

#		if self.separable:
#			self._H = tuple(self._check_struct(f,struct) 
#				for (f,struct) in zip(self._H,self.structure))
#		else:
#			struct0,struct = self.structure
#			assert struct0 is None
#			assert struct in ["scalar","matrix",None]

	@property
	def is_separable(self):
		return isinstance(self._H,tuple)
	@property
	def is_metric(self):
		return isinstance(self._H,Base)

	@property
	def vdim(self):
		"""
		Dimension of space of positions. 
		(Also equals the dimension of the space of impulsions)
		"""
		assert len(self.shape_free)==1
		return self.shape_free[0]

	@vdim.setter
	def vdim(self,vdim):
		self.shape_free = (vdim,)	

	def separable_quadratic_set_sparse_matrices():
		"""
		If the hamiltonian is separable and quadratic, replace the callable functions
		with sparse matrices, for faster evaluation.
		"""
		x_ad = ad.Sparse2.identity(shape=self.shape_free)
		self._H = tuple(scipy.sparse.coo_matrix(f(x_ad).triplets()).tocsc()
			if callable(f) else f
			for f in self._H)

#	def _check_struct(f,struct):
		"""
		Checks that the announced structure is present. 
		In the sparse case, will produce the required matrix from a callable using 
		automatic differentiation.
		- output : f, possibly turned into a matrix in the sparse case
		"""
#		if struct=="sparse" and callable(f):
#
#		if struct is None:
#			assert callable(f)
#		elif struct=="scalar":
#			assert np.isscalar(f)
#		elif struct=="matrix":
#			assert f.shape[:2]==shape_free*2
#		return f


#	@staticmethod
#	def _value(f,x):
		"""
		Used in separable case
		Evaluates a function with the given structure.
		"""
#		if callable(f): return f(x)
#		else: return 0.5*lp.dot_VV(x,ad.apply_linear_mapping(f,x))
#		if struct is None:
#			return f(x)
#		elif struct == "scalar":
#			return lo(0.5*f)*(x**2).sum()
#		elif struct == "matrix":
#			return 0.5*lp.dot_VAV(x,f,x)
#		elif struct == "sparse":
#			return 0.5*lp.dot_VV(x,ad.apply_linear_mapping(f,x))
#		raise "ValueError : unrecognized struct"

	def H(q,p):
		"""
		Evaluates the Hamiltonian, for a given position and impulsion.
		"""
		if self.is_separable:
			def value(f,x):
				"""Evaluates a function with the given structure."""
				if callable(f): return f(x)
				else: return 0.5*lp.dot_VV(x,ad.apply_linear_mapping(f,x))

			return sum(lo(value(h,x)) for (h,x) in zip(self._H,(q,p)) )
		elif self.is_metric:
			return self._H.at(q).norm2(p)
		else:
			return self._H(q,p)

#			_,struct = self.structure
#			if struct is None:
#				return self._H(q,p)
#			else:
#				h = self._H(q)
#				return self._value(h,p,struct)

	def _gradient(f,x):
		"""
		Differentiates a function with the given structure.
		"""
		if callable(f):
			x_ad = ad.Dense.identity(constant=x,shape_free=self.shape_free) 
			return f(x_ad).gradient()
		else:
			return ad.apply_linear_mapping(f,x)
#		if struct is None:
#			x_ad = ad.Dense.identity(constant=x,shape_free=self.shape_free) 
#			return f(x_ad).gradient()
#		elif struct == "scalar":
#			return lo(f)*x
#		elif struct == "matrix":
#			return lp.dot_AV(f,x)
#		elif struct == "sparse":
#			return ad.apply_linear_mapping(f,x)

	def DqH(q,p):
		"""
		Differentiates the Hamiltonian, w.r.t. position.
		"""
		if self.is_separable:
			return self._gradient(self._H[0],q)

		q_ad = ad.Dense.identity(constant=q,shape_free=self.shape_free)
		if self.is_metric:
			return self._H.at(q_ad).norm2(p).gradient()
		else: 
			return self._H(q_ad,p).gradient()

	def DpH(q,p):
		"""
		Differentiates the Hamiltonian, w.r.t. impulsion.
		"""
		if self.is_separable:
			return self._gradient(self._H[1],p)
		elif self.is_metric:
			return self._H.at(q).gradient2(p)
		else: 
			p_ad = ad.Dense.identity(constant=p,shape_free=self.shape_free)
			return self._H(q,p_ad).gradient()


#		if self.is_separable:
#			return self._gradient(self._H[1],p,self.structure[1],
#				shape_free=self.shape_free)
#		else:
#			_,struct = self.structure
#			if struct is None:
#				p_ad = ad.Dense.identity(constant=p,shape_free=self.shape_free)
#				return self._H(q,p_ad).gradient()
#			else:
#				h = self._H(q)
#				return self._gradient(h,p,struct)

	def flow(q,p):
		"""
		Symplectic flow of the Hamiltonian.
		"""
		return (DpH(q,p),-DqH(q,p))

	def nonsymplectic_schemes(self):
		"""
		Standard ODE integration schemes
		"""
		def Euler(q,p,dt):
			dq,dp = self.flow(q,p)
			return (q+dt*dq, p+dt*dp)
		def RK2(q,p,dt):
			dq1,dp1 = self.flow(q, p)
			dq2,dp2 = self.flow(q+0.5*dt*dq1, p+0.5*dt*dp1)
			return q+dt*dq2,p+dt*dp2
		def RK4(q,p,dt):
			dq1,dp1 = self.flow(q, p)
			dq2,dp2 = self.flow(q+0.5*dt*dq1, p+0.5*dt*dp1)
			dq3,dp3 = self.flow(q+0.5*dt*dq2, p+0.5*dt*dp2)
			dq4,dp4 = self.flow(q+dt*dq3, p+dt*dp3)
			return q+dt*(dq1+2*dq2+2*dq3+dq4)/6., p+dt*(dp1+2*dp2+2*dp3+dp4)/6.
		return {"Euler":Euler,"Runge-Kutta-2":RK2,"Runge-Kutta-4":RK4}

	def incomplete_schemes(self,solver=None):
		"""
		Incomplete schemes, updating only position or impulsion. 
		Inputs : 
			- solver (optional). Numerical solver used for the implicit steps.
			Defaults to a basic fixed point solver: "fixedpoint" in the same package.
		"""
		def Expl_q(q,p,dt):
			dq = self.DpH(q,p)
			return q+dt*dq,p
		def Expl_p(q,p,dt):
			dp = -self.DqH(q,p)
			return q,p+dt*dp

		if self.is_separable:
			return {"Explicit-q":Expl_q,"Explicit-p":Expl_p,
			"Implicit-q":Expl_q,"Implicit-p":Expl_p}

		if solver is None:
			solver=fixedpoint

		def Impl_q(q,p,dt):
			def f(q_): return q+dt*self.DpH(q_,p)
			return solver(f,q)
		def Impl_p(q,p,dt):
			def f(p_): return p-dt*self.DqH(q,p_)
			return solver(f,p)
		
		return {"Explicit-q":Expl_q,"Explicit-p":Expl_p,
			"Implicit-q":Impl_q,"Implicit-p":Impl_p}

	def symplectic_schemes(self,**kwargs):
		"""
		Symplectic schemes, alternating implicit and explicit updates, 
		of position and impulsion.
		The schemes become fully explicit in the case of a separable Hamiltonian.
		Inputs : 
			- kwargs. Passed to self.incomplete_schemes
		"""
		incomp = self.incomplete_schemes(**kwargs)
		Expl_q,Expl_p,Impl_q,Impl_p = tuple(incomp[s] for s in 
			"Explicit-q","Explicit-p","Implicit-q","Implicit-p")

		def Euler_pq(q,p,dt):
			q,p = Impl_p(q,p,dt)
			q,p = Expl_q(q,p,dt)
			return q,p

		def Euler_qp(q,p,dt):
			q,p = Impl_q(q,p,dt)
			q,p = Expl_p(q,p,dt)
			return q,p

		def Verlet_pqqp(q,p,dt):
			q,p = Impl_p(q,p,dt/2.)
			q,p = Expl_q(q,p,dt/2.)
			q,p = Impl_q(q,p,dt/2.)
			q,p = Expl_p(q,p,dt/2.)
			return q,p

		def Verlet_qppq(q,p,dt):
			q,p = Impl_q(q,p,dt/2.)
			q,p = Expl_p(q,p,dt/2.)
			q,p = Impl_p(q,p,dt/2.)
			q,p = Expl_q(q,p,dt/2.)
			return q,p

		return {"Euler-pq":Euler_pq,"Euler-qp":Euler_qp,
		"Verlet-pqqp":Verlet_pqqp,"Verlet-qppq":Verlet_qppq}













