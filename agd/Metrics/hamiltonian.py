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
lo = ad.left_operand

class Hamiltonian(object):
	def __init__(self, H, structure=(None,None), shape_free=None):
		"""
		Inputs:
		 - H : the hamiltonian, which may be either:
		 	* a callable function
			* a pair of callable functions, for a separable hamiltonian
			* a metric
		 	paira callable function. 
		   Separable hamiltonians can be specified as a pair.
		 - Structure : the hamiltonian structure, w.r.t. the first and second variable.
		   Possibilities include "scalar","matrix","sparse" and None
		"""
		is isinstance(H,list): H = tuple(H)
		self._H = H
		self.separable = isinstance(H,tuple)
		if self.separable: assert len(H)==2

		assert isinstance(structure,tuple) and len(structure)==2
		for e in structure: 
			assert e in ["scalar","matrix","sparse",None]
		self.structure = structure
		self.shape_free = shape_free

		if self.separable:
			self._H = tuple(self._check_struct(f,struct) 
				for (f,struct) in zip(self._H,self.structure))
		else:
			struct0,struct = self.structure
			assert struct0 is None
			assert struct in ["scalar","matrix",None]

	def _check_struct(f,struct):
		"""
		Checks that the announced structure is present. 
		In the sparse case, will produce the required matrix from a callable using 
		automatic differentiation.
		- output : f, possibly turned into a matrix in the sparse case
		"""
		if struct=="sparse" and callable(f):
			x_ad = ad.Sparse2.identity(shape=self.shape_free)
			f_ad = f(x_ad)
			return scipy.sparse.coo_matrix(f_ad.triplets()).tocsc()

		if struct is None:
			assert callable(f)
		elif struct=="scalar":
			assert np.isscalar(f)
		elif struct=="matrix":
			assert f.shape[:2]==shape_free*2
		return f


	@staticmethod
	def _value(f,x,struct):
		"""
		Evaluates a function with the given structure.
		"""
		if struct is None:
			return f(x)
		elif struct == "scalar":
			return lo(0.5*f)*(x**2).sum()
		elif struct == "matrix":
			return 0.5*lp.dot_VAV(x,f,x)
		elif struct == "sparse":
			return 0.5*lp.dot_VV(x,ad.apply_linear_mapping(f,x))
		raise "ValueError : unrecognized struct"

	def _gradient(f,x,struct):
		"""
		Differentiates a function with the given structure.
		"""
		if struct is None:
			x_ad = ad.Dense.identity(constant=x,shape_free=self.shape_free) 
			return f(x_ad).gradient()
		elif struct == "scalar":
			return lo(f)*x
		elif struct == "matrix":
			return lp.dot_AV(f,x)
		elif struct == "sparse":
			return ad.apply_linear_mapping(f,x)

	def H(q,p):
		"""
		Evaluates the Hamiltonian, for a given position and impulsion.
		"""
		if self.separable:
			return sum(lo(self._value(h,x,s))
				for (h,x,s) in zip(self._H,(q,p),self.structure))
		else:
			_,struct = self.structure
			if struct is None:
				return self._H(q,p)
			else:
				h = self._H(q)
				return self._value(h,p,struct)

	def DqH(q,p):
		"""
		Differentiates the Hamiltonian, w.r.t. position.
		"""
		if self.separable:
			return self._deriv(self._H[0],q,self.structure[0])
		else:
			q_ad = ad.Dense.identity(constant=q,shape_free=self.shape_free)
			return self._H(q_ad,p).gradient()

	def DpH(q,p):
		"""
		Differentiates the Hamiltonian, w.r.t. impulsion.
		"""
		if self.separable:
			return self._gradient(self._H[1],p,self.structure[1],
				shape_free=self.shape_free)
		else:
			_,struct = self.structure
			if struct is None:
				p_ad = ad.Dense.identity(constant=p,shape_free=self.shape_free)
				return self._H(q,p_ad).gradient()
			else:
				h = self._H(q)
				return self._gradient(h,p,struct)

	def flow(q,p):
		"""
		Symplectic flow of the Hamiltonian
		"""
		return (DpH(q,p),-DqH(q,p))

	def set_scheme(dt,scheme,solver=None):
		"""
		Set the scheme for numerical integration.
		inputs : 
			- dt : time step
			- scheme : choice of ode integration scheme
		"""
		pass

#	def step(q,p):
#		if scheme==












