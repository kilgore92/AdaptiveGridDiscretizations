import numpy as np

# ------- Ugly utilities -------
def _tuple_first(a): 	return a[0] if isinstance(a,tuple) else a
def _getitem(a,where):
	return a if where is True else a[where]
def _add_dim(a):		return np.expand_dims(a,axis=-1)	
def _add_dim2(a):		return _add_dim(_add_dim(a))
def _pad_last(a,pad_total): # Always makes a deep copy
		return np.pad(a, pad_width=((0,0),)*(a.ndim-1)+((0,pad_total-a.shape[-1]),), mode='constant', constant_values=0)
def _add_coef(a,b):
	if a.shape[-1]==0: return b
	elif b.shape[-1]==0: return a
	else: return a+b
def _prep_nl(s): return "\n"+s if "\n" in s else s
def _concatenate(a,b): 	return np.concatenate((a,b),axis=-1)

def _set_shape_free_bound(shape,shape_free,shape_bound):
	if shape_free is not None:
		assert shape_free==shape[0:len(shape_free)]
		if shape_bound is None: 
			shape_bound=shape[len(shape_free):]
		else: 
			assert shape_bound==shape[len(shape_free):]
	if shape_bound is None: 
		shape_bound = tuple()
	assert len(shape_bound)==0 or shape_bound==shape[-len(shape_bound):]
	if shape_free is None:
		if len(shape_bound)==0:
			shape_free = shape
		else:
			shape_free = shape[:len(shape_bound)]
	return shape_free,shape_bound

def _set_shape_constant(shape=None,constant=None):
	if constant is None:
		if shape is None:
			raise ValueError("Error : unspecified shape or constant")
		constant = np.full(shape,0.)
	else:
		if not isinstance(constant,np.ndarray):
			constant = np.array(constant)
		if shape is not None and shape!=constant.shape: 
			raise ValueError("Error : incompatible shape and constant")
		else:
			shape=constant.shape
	return shape,constant

def _test_or_broadcast_ad(array,shape,broadcast,ad_depth=1):
	if broadcast:
		if array.shape[:-ad_depth]==shape:
			return array
		else:
			return np.broadcast_to(array,shape+array.shape[-ad_depth:])
	else:
		assert array.shape[:-ad_depth]==shape
		return array

# -------- For Dense and Dense2 -----

def apply_linear_operator(op,rhs,flatten_ndim=0):
	"""Applies a linear operator to an array with more than two dimensions,
	by flattening the last dimensions"""
	assert (rhs.ndim-flatten_ndim) in [1,2]
	shape_tail = rhs.shape[1:]
	op_input = rhs.reshape((rhs.shape[0],np.prod(shape_tail)))
	op_output = op(op_input)
	return op_output.reshape((op_output.shape[0],)+shape_tail)


# -------- For Reverse and Reverse2 -------

# Applying a function
def _apply_output_helper(rev,a):
	"""
	Adds 'virtual' AD information to an output (with negative indices), 
	in selected places.
	"""
	from . import is_ad
	import numbers
	assert not is_ad(a)
	if isinstance(a,tuple): 
		result = tuple(_apply_output_helper(x) for x in a)
		return tuple(x for x,_ in result), tuple(y for _,y in result)
	elif isinstance(a,np.ndarray) and not issubclass(a.dtype.type,numbers.Integral):
		shape = [rev.size_rev,a.shape]
		return rev._identity_rev(constant=a),shape
	else:
		return a,None

def _apply_input_helper(args,kwargs,cls):
	"""
	Removes the AD information from some function input, and provides the correspondance.
	"""
	from . import is_ad
	corresp = []
	def _make_arg(a):
		nonlocal corresp
		if is_ad(a):
			assert isinstance(a,cls)
			a_value = np.array(a)
			corresp.append((a,a_value))
			return a_value
		else:
			return a
	_args = [_make_arg(a) for a in args]
	_kwargs = {key:_make_arg(val) for key,val in kwargs.items()}
	return _args,_kwargs,corresp


def _to_shapes(coef,shapes):
	"""
	Reshapes a one dimensional array into the given shapes, 
	given as a tuple of [start,shape]
	""" 
	if shapes is None: 
		return None
	elif isinstance(shapes,tuple): 
		return tuple(_to_shapes(coef,s) for s in shapes)
	else:
		start,shape = shapes
		return coef[start : start+np.prod(shape)].reshape(shape)

# ----- Functionnal -----

def recurse(step,niter=1):
	def operator(rhs):
		nonlocal step,niter
		for i in range(niter):
			rhs=step(rhs)
		return rhs
	return operator

# ------- Common functions -------

def spsolve(mat,rhs):
	"""
	Solves a sparse linear system where the matrix is given as triplets.
	"""
	import scipy.sparse; import scipy.sparse.linalg
	return scipy.sparse.linalg.spsolve(
	scipy.sparse.coo_matrix(mat).tocsr(),rhs)

def spapply(mat,rhs,crop_rhs=False):
	"""
	Applies a sparse matrix, given as triplets, to an rhs.
	"""
	import scipy.sparse; import scipy.sparse.linalg
	if crop_rhs: 
		cols = mat[1][1]
		if len(cols)==0: 
			return np.zeros(0)
		size = 1+np.max(cols)
		if rhs.shape[0]>size:
			rhs = rhs[:size]
	return scipy.sparse.coo_matrix(mat).tocsr()*rhs


def min(array,axis=None,keepdims=False,out=None):
	if axis is None: return array.flatten().min(axis=0,out=out)
	ai = np.expand_dims(np.argmin(array.value, axis=axis), axis=axis)
	out = np.take_along_axis(array,ai,axis=axis)
	if not keepdims: out = out.reshape(array.shape[:axis]+array.shape[axis+1:])
	return out

def max(array,axis=None,keepdims=False,out=None):
	if axis is None: return array.flatten().max(axis=0,out=out)
	ai = np.expand_dims(np.argmax(array.value, axis=axis), axis=axis)
	out = np.take_along_axis(array,ai,axis=axis)
	if not keepdims: out = out.reshape(array.shape[:axis]+array.shape[axis+1:])
	return out

def add(a,b,out=None,where=True): 
	from . import is_ad
	if out is None: return a+b if is_ad(a) else b+a
	else: result=_tuple_first(out); result[where]=a[where]+_getitem(b,where); return result

def subtract(a,b,out=None,where=True):
	from . import is_ad
	if out is None: return a-b if is_ad(a) else b.__rsub__(a) 
	else: result=_tuple_first(out); result[where]=a[where]-_getitem(b,where); return result

def multiply(a,b,out=None,where=True): 
	from . import is_ad
	if out is None: return a*b if is_ad(a) else b*a
	else: result=_tuple_first(out); result[where]=a[where]*_getitem(b,where); return result

def true_divide(a,b,out=None,where=True): 
	from . import is_ad
	if out is None: return a/b if is_ad(a) else b.__rtruediv__(a)
	else: result=_tuple_first(out); result[where]=a[where]/_getitem(b,where); return result


def maximum(a,b): 	
	from . import where
	return where(a>b,a,b)
def minimum(a,b): 	
	from . import where
	return where(a<b,a,b)