import numpy as np

# ------- Ugly utilities -------
def _tuple_first(a): 	return a[0] if isinstance(a,tuple) else a
def _getitem(a,where):
	return a if where is True else a[where]

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


