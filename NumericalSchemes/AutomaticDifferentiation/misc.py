import numpy as np

# ------- Ugly utilities -------
def _tuple_first(a): 	return a[0] if isinstance(a,tuple) else a
def _getitem(a,where):
	return a if where is True else a[where]

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

#def _broadcast_coef(t,dense2=False):
#	shape = t[0].shape
#	def broadcast(array,i=1):
#		if array.shape[:-i]==shape: return array
#		else: return np.broadcast_to(array,shape+array.shape[-i:])
#	if dense2: return (t[0], broadcast(t[1]), broadcast(t[2],2))
#	else: return (t[0],)+tuple(broadcast(ti) for ti in t[1:])

# ------- Common functions -------

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