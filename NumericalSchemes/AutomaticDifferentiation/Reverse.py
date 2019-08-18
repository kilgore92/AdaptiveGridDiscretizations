import numpy as np
import copy
from . import misc
from . import Dense
from . import Sparse

class reverseAD(object):
	"""
	A class for reverse first order automatic differentiation
	"""

	def __init__(self):
		self.deepcopy_states = False
		self._size_ad = 0
		self._size_rev = 0
		self._states = []
		self._shapes_ad = tuple()

	@property
	def size_ad(self): return self._size_ad
	@property
	def size_rev(self): return self._size_rev

	# Variable creation
	def identity(self,*args,**kwargs):
		result = Sparse.identity(*args,**kwargs,shift=self.size_ad)
		self._shapes_ad += ([self.size_ad,result.shape],)
		self._size_ad += result.size
		return result

	def _identity_rev(self,*args,**kwargs):
		result = Sparse.identity(*args,**kwargs,shift=self.size_rev)
		self._size_rev += result.size
		result.index = -result.index-1
		return result

	def _index_rev(self,index):
		index=index.copy()
		pos = index<0
		index[pos] = -index[pos]-1+self.size_ad
		return index

	# Applying a function

	def _apply_output_helper(self,a):
		"""
		Adds 'virtual' AD information to an output (with negative indices), 
		in selected places.
		"""
		from . import is_ad
		import numbers
		assert not is_ad(a)
		if isinstance(a,tuple): 
			result = tuple(self._apply_output_helper(x) for x in a)
			return tuple(x for x,_ in result), tuple(y for _,y in result)
		elif isinstance(a,np.ndarray) and not issubclass(a.dtype.type,numbers.Integral):
			shape = [self.size_rev,a.shape]
			return self._identity_rev(constant=a),shape
		else:
			return a,None

	@staticmethod
	def _apply_input_helper(args,kwargs):
		"""
		Removes the AD information from some function input, and provides the correspondance.
		"""
		from . import is_ad
		corresp = []
		def _make_arg(a):
			nonlocal corresp
			if is_ad(a):
				assert isinstance(a,Sparse.spAD)
				a_value = np.array(a)
				corresp.append((a,a_value))
				return a_value
			else:
				return a
		_args = [_make_arg(a) for a in args]
		_kwargs = {key:_make_arg(val) for key,val in kwargs.items()}
		return _args,_kwargs,corresp

	def apply(self,func,*args,**kwargs):
		"""
		Applies a function on the given args, saving adequate data
		for reverse AD.
		"""
		_args,_kwargs,corresp = reverseAD._apply_input_helper(args,kwargs)
		if len(corresp)==0: return f(args,kwargs)
		_output = func(*_args,**_kwargs)
		output,shapes = self._apply_output_helper(_output)
		self._states.append((shapes,func,
			copy.deepcopy(args) if self.deepcopy_states else args,
			copy.deepcopy(kwargs) if self.deepcopy_states else kwargs))
		return output

	def apply_linear_mapping(self,matrix,rhs,niter=0):
		return self.apply(linear_mapping_with_adjoint(matrix,niter=niter),rhs)
	def apply_linear_inverse(self,matrix,solver,rhs,niter=0):
		return self.apply(linear_inverse_with_adjoint(matrix,solver,niter=niter),rhs)
	def apply_identity(self,rhs):
		return self.apply(identity_with_adjoint,rhs)

	def iterate(self,func,var,*args,**kwargs):
		"""
		Input: function, variable to be updated, niter, nrec, optional args
		Iterates a function, saving adequate data for reverse AD. 
		If nrec>0, a recursive strategy is used to limit the amount of data saved.
		"""
		niter = kwargs.pop('niter')
		nrec = 0 if niter<=1 else kwargs.pop('nrec',0)
		assert nrec>=0
		if nrec==0:
			for i in range(niter):
				var = self.apply(func,
					var if self.deepcopy_states else copy.deepcopy(var),
					*args,**kwargs)
			return var
		else:
			assert False #TODO
		"""
			def recursive_iterate():
				other = reverseAD()
				return other.iterate(func,
			niter_top = int(np.ceil(niter**(1./(1+nrec))))
			for rec_iter in (niter//niter_top,)*niter_top + (niter%niter_top,)
				
				var = self.apply(recursive_iterate,var,*args,**kwargs,niter=rec_iter,nrec=nrec-1)

		for 
		"""


	# Adjoint evaluation pass
	@staticmethod
	def _to_shapes(coef,shapes):
		if shapes is None: 
			return None
		elif isinstance(shapes,tuple): 
			return tuple(reverseAD._to_shapes(coef,s) for s in shapes)
		else:
			start,shape = shapes
			return coef[start : start+np.prod(shape)].reshape(shape)

	def gradient(self,a):
		coef = Sparse.spAD(a.value,a.coef,self._index_rev(a.index)).to_dense().coef
		for outputshapes,func,args,kwargs in reversed(self._states):
			co_output = reverseAD._to_shapes(coef[self.size_ad:],outputshapes)
			_args,_kwargs,corresp = reverseAD._apply_input_helper(args,kwargs)
			co_args = func(*_args,**_kwargs,co_output=co_output)
			for a_adjoint,a_value in co_args:
				for a_sparse,a_value2 in corresp:
					if a_value is a_value2:
						val,(row,col) = a_sparse.triplets()
						coef_contrib = misc.spapply(
							(val,(self._index_rev(col),row)),
							a_adjoint)
						# Possible improvement : shift by np.min(self._index_rev(col)) to avoid adding zeros
						coef[:coef_contrib.shape[0]] += coef_contrib
						break
		return coef[:self.size_ad]

	def to_inputshapes(self,a):
		return reverseAD._to_shapes(a,self._shapes_ad)
# End of class reverseAD

def empty():
	return reverseAD()

# Elementary operators with adjoints

def linear_inverse_with_adjoint(matrix,solver,niter=1):
	def operator(u,co_output=None):
		nonlocal matrix,solver,niter
		if co_output is None:
			for i in range(niter):
				u=solver(matrix,u)
			return u
		else:
			for i in range(niter):
				co_output = solver(matrix.T,co_output)
			return [(co_output,u)]
	return operator

def linear_mapping_with_adjoint(matrix,niter=1):
	def operator(u,co_output=None):
		nonlocal matrix
		if co_output is None:
			for i in range(niter):
				u=matrix*u
			return u
		else:
			for i in range(niter):
				co_output = matrix.T*co_output
			return [(co_output,u)]
	return operator

def identity_with_adjoint(u,co_output=None):
	if co_output is None:
		return u
	else:
		return [(co_output,u)]
