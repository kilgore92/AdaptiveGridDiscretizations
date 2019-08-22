from . import misc
from . import Dense
from . import Sparse
from . import Reverse
from . import Dense2
from . import Sparse2
from . import Reverse2
import numpy as np
import itertools

def reload_submodules():
	import importlib
	import sys
	ad = sys.modules['NumericalSchemes.AutomaticDifferentiation']
	ad.misc = importlib.reload(ad.misc)
	ad.Dense = importlib.reload(ad.Dense)
	ad.Sparse = importlib.reload(ad.Sparse)
	ad.Reverse = importlib.reload(ad.Reverse)
	ad.Sparse2 = importlib.reload(ad.Sparse2)
	ad.Dense2 = importlib.reload(ad.Dense2)
	ad.Reverse2 = importlib.reload(ad.Reverse2)


def is_adtype(t):
	return t in (Sparse.spAD, Dense.denseAD, Sparse2.spAD2, Dense2.denseAD2)

def is_ad(array):
	return is_adtype(type(array)) 

def simplify_ad(a):
	if type(a) in (Sparse.spAD,Sparse2.spAD2): 
		a.simplify_ad()

def is_strict_subclass(type0,type1):
	return issubclass(type0,type1) and type0!=type1

def toarray(a,array_type=np.ndarray):
	if isinstance(a,array_type): return a
	return array_type(a) if is_strict_subclass(array_type,np.ndarray) else np.array(a)

def broadcast_to(array,shape):
	if is_ad(array): return array.broadcast_to(shape)
	else: return np.broadcast_to(array,shape)

def where(mask,a,b): 
	if is_ad(a) or is_ad(b):
		A,B,Mask = (a,b,mask) if is_ad(b) else (b,a,np.logical_not(mask))
		result = B.copy()
		result[Mask] = A[Mask] if isinstance(A,np.ndarray) else A
		return result
	else: 
		return np.where(mask,a,b)

def sort(array,axis=-1,*varargs,**kwargs):
	if is_ad(array):
		ai = np.argsort(array.value,axis=axis,*varargs,**kwargs)
		return np.take_along_axis(array,ai,axis=axis)
	else:
		return np.sort(array,axis=axis,*varargs,**kwargs)

def min_argmin(array,axis=None):
	if axis is None: return min_argmin(array.flatten(),axis=0)
	ai = np.argmin(array,axis=axis)
	return np.squeeze(np.take_along_axis(array,np.expand_dims(ai,
		axis=axis),axis=axis),axis=axis),ai

def max_argmax(array,axis=None):
	if axis is None: return max_argmax(array.flatten(),axis=0)
	ai = np.argmax(array,axis=axis)
	return np.squeeze(np.take_along_axis(array,np.expand_dims(ai,
		axis=axis),axis=axis),axis=axis),ai

def stack(elems,axis=0):
	for e in elems:
		if is_ad(e): return type(e).stack(elems,axis)
	return np.stack(elems,axis)

def concatenate(elems,axis=0):
	for e in elems:
		if is_ad(e): return type(e).concatenate(elems,axis)
	return np.concatenate(elems,axis)

def disassociate(array,shape_free=None,shape_bound=None,singleton_axis=-1):
	shape_free,shape_bound = misc._set_shape_free_bound(array.shape,shape_free,shape_bound)
	size_free = np.prod(shape_free)
	array = array.reshape((size_free,)+shape_bound)
	result = np.zeros(size_free,object)
	for i in range(size_free): result[i] = array[i]
	result = result.reshape(shape_free)
	if singleton_axis is not None:
		result=np.expand_dims(result,singleton_axis)
	return result

def associate(array,singleton_axis=-1):
	if is_ad(array): 
		return array.associate(singleton_axis)
	result = stack(array.flatten(),axis=0)
	shape_free = array.shape
	if singleton_axis is not None: 
		assert shape_free[singleton_axis]==1
		if singleton_axis==-1:
			shape_free=shape_free[:-1]
		else: 
			shape_free=shape_free[:singleton_axis]+shape_free[(singleton_axis+1):]
	return result.reshape(shape_free+result.shape[1:])

def apply(f,*args,**kwargs):
	envelope,shape_bound,reverse_history = (kwargs.pop(s,None) for s in ('envelope','shape_bound','reverse_history'))
	if not any(is_ad(a) for a in itertools.chain(args,kwargs.values())):
		return f(*args,**kwargs)
	if envelope:
		def to_np(a): return a.value if is_ad(a) else a
		_,oracle = f(*[to_np(a) for a in args],**{key:to_np(val) for key,val in kwargs.items()})
		result,_ = apply(f,*args,**kwargs,oracle=oracle,envelope=False,shape_bound=shape_bound)
		return result,oracle
	if shape_bound:
		size_factor = np.prod(shape_bound)
		t = tuple(b.reshape((b.size//size_factor,)+shape_bound) 
			for b in itertools.chain(args,kwargs.values()) if is_ad(b)) # Tuple containing the original AD vars
		lens = tuple(len(b) for b in t)
		def to_dense(b):
			if not is_ad(b): return b
			nonlocal i
			shift = (sum(lens[:i]),sum(lens[(i+1):]))
			i+=1
			if type(b) in (Sparse.spAD,Dense.denseAD): 
				return Dense.identity(constant=b.value,shape_bound=shape_bound,shift=shift)
			elif type(b) in (Sparse2.spAD2,Dense2.denseAD2):
				return Dense2.identity(constant=b.value,shape_bound=shape_bound,shift=shift)
		i=0
		args2 = [to_dense(b) for b in args]
		kwargs2 = {key:to_dense(val) for key,val in kwargs.items()}
		result2 = f(*args2,**kwargs2)
		return compose(result2,t,shape_bound=shape_bound)
	if reverse_history:
		return reverse_history.apply(f,*args,**kwargs)
	return f(*args,**kwargs)

def compose(a,t,shape_bound):
	"""Compose ad types, mostly intended for dense a and sparse b"""
	if not isinstance(t,tuple): t=(t,)
	if isinstance(a,tuple):
		return tuple(compose(ai,t,shape_bound) for ai in a)
	if not(type(a) in (Dense.denseAD,Dense2.denseAD2)) or len(t)==0:
		return a
	return type(t[0]).compose(a,t)

def apply_linear_mapping(matrix,rhs,niter=1):
	def step(x):
		nonlocal matrix
		return np.dot(matrix,x) if isinstance(matrix,np.ndarray) else (matrix*x)
	operator = misc.recurse(step,niter)
	return rhs.apply_linear_operator(operator) if is_ad(rhs) else operator(rhs)

def apply_linear_inverse(solver,matrix,rhs,niter=1):
	def step(x):
		nonlocal solver,matrix
		return solver(matrix,x)
	operator = misc.recurse(step,niter)
	return rhs.apply_linear_operator(operator) if is_ad(rhs) else operator(rhs)

"""
	if isinstance(a,Dense.denseAD) and (isinstance(b,Sparse.spAD) or all(isinstance(e,Sparse.spAD) for e in b)):
		elem = None
		size_factor = np.prod(shape_bound)
		if shape_bound is None:
			if not isinstance(b,Sparse.spAD):
				raise ValueError("Compose error : unspecified shape_bound")
			elem = b
		elif isinstance(b,Sparse.spAD):
			elem = b.reshape( (b.size//size_factor,)+shape_bound)
		else:
			elem = stack(e.reshape( (e.size//size_factor,)+shape_bound) for e in b)

		if elem.shape[0]!=a.size_ad:
			raise ValueError("Compose error : incompatible shapes")
		coef = np.moveaxis(a.coef,-1,0)
		first_order = sum(x*y for x,y in zip(coef,elem))
		return Sparse.spAD(a.value,first_order.coef,first_order.index)
	else:
		raise ValueError("Only Dense-Sparse composition is implemented")

def dense_eval(f,b,shape_bound):
	if isinstance(b,Sparse.spAD):
		b_dense = Dense.identity(b.shape,shape_bound,constant=b)
		return compose(f(b_dense),b,shape_bound=shape_bound)
	elif all(isinstance(e,Sparse.spAD) for e in b):
		size_factor = np.prod(shape_bound)
		size_ad_all = tuple(e.size/size_factor for e in b)
		size_ad = sum(size_ad_all)
		size_ad_cumsum = np.cumsum(size_ad_all)
		size_ad_cumsum=(0,)+size_ad_cumsum[:-1]
		size_ad_revsum = np.cumsum(reversed(size_ad_all))
		size_ad_revsum=(0,)+size_ad_revsum[:-1] 

		b_dense = stack(tuple(
			Dense.identity(e.shape,shape_bound,constant=e,padding=(padding_before,padding_after))
				for e,padding_before,padding_after in zip(b,size_ad_cumsum,size_ad_revsum) 
				))
		return compose(f(b_dense),b,shape_bound=shape_bound)
	else:
		return f(b)
"""


