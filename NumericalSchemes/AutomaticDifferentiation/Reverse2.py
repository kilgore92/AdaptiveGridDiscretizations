import numpy as np
import copy
from . import misc
from . import Dense
from . import Sparse
from . import Reverse
from . import Sparse2

class reverseAD2(object):
	"""
	A class for reverse second order automatic differentiation
	"""

	def __init__(self,operator_data=None):
		self.operator_data=operator_data
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
		"""Creates and register a new AD variable"""
		result = Sparse2.identity(*args,**kwargs,shift=self.size_ad)
		self._shapes_ad += ([self.size_ad,result.shape],)
		self._size_ad += result.size
		return result

	def _identity_rev(self,*args,**kwargs):
		"""Creates and register an AD variable with negative indices, 
		used as placeholders in reverse AD"""
		result = Sparse2.identity(*args,**kwargs,shift=self.size_rev)
		self._size_rev += result.size
		result.index = -result.index-1
		return result

	def _index_rev(self,index):
		"""Turns the negative placeholder indices into positive ones, 
		for sparse matrix creation."""
		index=index.copy()
		pos = index<0
		index[pos] = -index[pos]-1+self.size_ad
		return index

	def apply(self,func,*args,**kwargs):
		"""
		Applies a function on the given args, saving adequate data
		for reverse AD.
		"""
		_args,_kwargs,corresp = misc._apply_input_helper(args,kwargs,Sparse2.spAD2)
		if len(corresp)==0: return f(args,kwargs)
		_output = func(*_args,**_kwargs)
		output,shapes = misc._apply_output_helper(self,_output)
		self._states.append((shapes,func,
			copy.deepcopy(args) if self.deepcopy_states else args,
			copy.deepcopy(kwargs) if self.deepcopy_states else kwargs))
		return output

	def apply_linear_mapping(self,matrix,rhs,niter=1):
		return self.apply(Reverse.linear_mapping_with_adjoint(matrix,niter=niter),rhs)
	def apply_linear_inverse(self,matrix,solver,rhs,niter=1):
		return self.apply(Reverse.linear_inverse_with_adjoint(matrix,solver,niter=niter),rhs)
	def simplify(self,rhs):
		return self.apply(Reverse.identity_with_adjoint,rhs)

	# Adjoint evaluation pass
	def gradient(self,a):
		coef = Sparse.spAD(a.value,a.coef1,self._index_rev(a.index)).to_dense().coef
		for outputshapes,func,args,kwargs in reversed(self._states):
			co_output = misc._to_shapes(coef[self.size_ad:],outputshapes)
			_args,_kwargs,corresp = misc._apply_input_helper(args,kwargs,Sparse2.spAD2)
			co_args = func(*_args,**_kwargs,co_output=co_output)
			for a_value,a_adjoint in co_args:
				for a_sparse,a_value2 in corresp:
					if a_value is a_value2:
						val,(row,col) = a_sparse.to_first().triplets()
						coef_contrib = misc.spapply(
							(val,(self._index_rev(col),row)),
							a_adjoint)
						# Possible improvement : shift by np.min(self._index_rev(col)) to avoid adding zeros
						coef[:coef_contrib.shape[0]] += coef_contrib
						break
		return coef[:self.size_ad]

	def _hessian_forward_input_helper(self,args,kwargs,dir):
		"""Replaces Sparse AD information with dense one, based on dir_hessian."""
		from . import is_ad
		corresp = []
		def _make_arg(a):
			nonlocal dir,corresp
			if is_ad(a):
				assert isinstance(a,Sparse2.spAD2)
				a1=Sparse.spAD(a.value,a.coef1,self._index_rev(a.index))
				coef = misc.spapply(a1.triplets(),dir[:a1.bound_ad()])
				a_value = Dense.denseAD(a1.value, coef.reshape(a.shape+(1,)))
				corresp.append((a,a_value))
				return a_value
			else:
				return a
		_args = [_make_arg(a) for a in args]
		_kwargs = {key:_make_arg(val) for key,val in kwargs.items()}
		return _args,_kwargs,corresp

	def _hessian_forward_make_dir(self,values,shapes,dir):
		if shapes is None: pass
		elif isinstance(shapes,tuple): 
			for value,shape in zip(values,shapes):
				_hessian_forward_make_dir(values,shapes)
		else:
			start,shape = shapes
			assert isinstance(values,Dense.denseAD) and values.size_ad==1
			assert values.shape==shape
			sstart = self.size_ad+start
			dir[sstart:(sstart+values.size)] = values.coef.flatten()

	def hessian(self,a):
		def hess_operator(dir_hessian,coef2_init=None,with_grad=False):
			nonlocal self,a
			# Forward pass : propagate the hessian direction
			size_total = self.size_ad+self.size_rev
			dir_hessian_forwarded = np.zeros(size_total)
			dir_hessian_forwarded[:self.size_ad] = dir_hessian
			denseArgs = []
			for outputshapes,func,args,kwargs in self._states:
				# Produce denseAD arguments containing the hessian direction
				_args,_kwargs,corresp = self._hessian_forward_input_helper(args,kwargs,dir_hessian_forwarded)
				denseArgs.append((_args,_kwargs,corresp))
				# Evaluate the function 
				output = func(*_args,**_kwargs)
				# Collect the forwarded hessian direction
				self._hessian_forward_make_dir(output,outputshapes,dir_hessian_forwarded)

			# Reverse pass : evaluate the hessian operator
			# TODO : avoid the recomputation of the gradient
			coef1 = Sparse.spAD(a.value,a.coef1,self._index_rev(a.index)).to_dense().coef
			coef2 = misc.spapply((a.coef2,(self._index_rev(a.index_row),self._index_rev(a.index_col))),dir_hessian_forwarded, crop_rhs=True)
			if coef1.size<size_total:  coef1 = misc._pad_last(coef1,size_total)
			if coef2.size<size_total:  coef2 = misc._pad_last(coef2,size_total)
			if not(coef2_init is None): coef2 += misc._pad_last(coef2_init,size_total)
			for (outputshapes,func,_,_),(_args,_kwargs,corresp) in zip(reversed(self._states),reversed(denseArgs)):
				co_output1 = misc._to_shapes(coef1[self.size_ad:],outputshapes)
				co_output2 = misc._to_shapes(coef2[self.size_ad:],outputshapes)
				co_args = func(*_args,**_kwargs,co_output=co_output1,co_output2=co_output2)
				for a_value,a_adjoint1,a_adjoint2 in co_args:
					for a_sparse,a_value2 in corresp:
						if a_value is a_value2:
							# Linear contribution to the gradient
							val,(row,col) = a_sparse.to_first().triplets()
							triplets = (val,(self._index_rev(col),row))
							coef1_contrib = misc.spapply(triplets,a_adjoint1)
							coef1[:coef1_contrib.shape[0]] += coef1_contrib

							# Linear contribution to the hessian
							linear_contrib = misc.spapply(triplets,a_adjoint2)
							coef2[:linear_contrib.shape[0]] += linear_contrib

							# Quadratic contribution to the hessian
							obj = (a_adjoint1*a_sparse).sum()
							quadratic_contrib = misc.spapply((obj.coef2,(self._index_rev(obj.index_row),self._index_rev(obj.index_col))), dir_hessian_forwarded, crop_rhs=True)
							coef2[:quadratic_contrib.shape[0]] += quadratic_contrib

							break
			return (coef1[:self.size_ad],coef2[:self.size_ad]) if with_grad else coef2[:self.size_ad]
		return hess_operator


	def to_inputshapes(self,a):
		return misc._to_shapes(a,self._shapes_ad)

	def output(self,a):
		assert not(self.operator_data is None)
		if self.operator_data is "PassThrough":
			return a
		inputs,co_output,co_output2,dir_hessian = self.operator_data
		_a = misc.sumprod(a,co_output)
		def sumprod2(u,v):
			if u is None: return 0.
			elif isinstance(u,tuple): return sum(sumprod(x,y) for (x,y) in zip(u,v))
			else: return (u.to_first()*v).sum()
		_a2 = sumprod2(a,co_output2)
		coef2_init = Sparse.spAD(_a2.value,self._index_rev(_a2.index)).to_dense().coef

		hess = self.hessian(_a)
		coef1,coef2 = hess(dir_hessian,coef2_init=coef2_init,with_grad=True)
		return [(x,y,z) for (x,y,z) in zip(inputs,self.to_inputshapes(coef1),self.to_inputshapes(coef2))]

# End of class reverseAD2

def empty():
	return reverseAD2()

def operator_like(inputs=None,co_output=None,co_output2=None):
	"""
	Operator_like reverseAD2 : 
	- should not register new inputs (conflicts with the way dir_hessian is provided)
	- fixed co_output and co_output2
	- gets dir_hessian from inputs
	"""
	if co_output is None: return reverseAD2(operator_data="PassThrough"),inputs
	elif co_output2 is None:
		from . import Reverse
		return Reverse.operator_like(inputs,co_output,co_output2)
	else:
		assert all(isinstance(a,Dense.denseAD) and a.size_ad==1 for a in inputs)
		dir_hessian = np.concatenate(tuple(a.coef.flatten() for a in inputs))
		rev = reverseAD2(operator_data=(inputs,co_output,co_output2,dir_hessian))
		_inputs = tuple(rev.identity(constant=a.value) for a in inputs)
		return rev,_inputs