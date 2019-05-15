from . import Sparse
import numpy as np

def is_adtype(t):
	return t==Sparse.spAD 

def is_ad(array):
	return is_adtype(type(array)) 

def is_strict_subclass(type0,type1):
	return issubclass(type0,type1) and type0!=type1

def toarray(a,array_type=np.ndarray):
	if isinstance(a,array_type): return a
	return array_type(a) if is_strict_subclass(array_type,np.ndarray) else np.array(a)

def broadcast_to(array,shape):
	if is_ad(array): return array.broadcast_to(shape)
	else: return np.broadcast_to(array,shape)

def where(mask,a,b): 
	if is_ad(b): return b.replace_at(mask,a) 
	elif is_ad(a): return a.replace_at(np.logical_not(mask),b) 
	else: return np.where(mask,a,b)

def sort(array,axis=-1,*varargs,**kwargs):
	if is_ad(array):
		ai = np.argsort(array.value,axis=axis,*varargs,**kwargs)
		return np.take_along_axis(array,ai,axis=axis)
	else:
		return np.sort(array,axis=axis,*varargs,**kwargs)

def stack(elems,axis=0):
	for e in elems:
		if is_ad(e): return type(e).stack(elems,axis)
	return np.stack(elems,axis)
