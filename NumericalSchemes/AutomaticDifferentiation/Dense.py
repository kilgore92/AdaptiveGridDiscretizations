import numpy as np
from . import misc

class denseAD(np.ndarray):
	"""
	A class for dense forward automatic differentiation
	"""

	# Construction
	# See : https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
	def __new__(cls,value,coef=None,broadcast_ad=False):
		if isinstance(value,denseAD):
			assert coef is None
			return value
		obj = np.asarray(value).view(denseAD)
		shape = obj.shape
		shape2 = shape+(0,)
		obj.coef  = (np.full(shape2,0.) if coef is None 
			else misc._test_or_broadcast_ad(coef,shape,broadcast_ad) )
		return obj

#	def __array_finalize__(self,obj): pass

	def copy(self,order='C'):
		return denseAD(self.value.copy(order=order),self.coef.copy(order=order))

	# Representation 
	def __iter__(self):
		for value,coef in zip(self.value,self.coef):
			yield denseAD(value,coef)

	def __str__(self):
		return "denseAD("+str(self.value)+","+_prep_nl(str(self.coef))+")"
#		return "denseAD"+str((self.value,self.coef))
	def __repr__(self):
		return "denseAD("+repr(self.value)+","+_prep_nl(repr(self.coef))+")"
#		return "denseAD"+repr((self.value,self.coef))	

	# Operators
	def __add__(self,other):
		if _is_constant(other): return self.__add__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value+other.value, _add_coef(self.coef,other.coef))
		else:
			return denseAD(self.value+other, self.coef, broadcast_ad=True)

	def __sub__(self,other):
		if _is_constant(other): return self.__sub__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value-other.value, _add_coef(self.coef,-other.coef))
		else:
			return denseAD(self.value-other, self.coef, broadcast_ad=True)

	def __mul__(self,other):
		if _is_constant(other): return self.__mul__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value*other.value,_add_coef(_add_dim(other.value)*self.coef,_add_dim(self.value)*other.coef))
		elif isinstance(other,np.ndarray):
			return denseAD(self.value*other,_add_dim(other)*self.coef)
		else:
			return denseAD(self.value*other,other*self.coef)

	def __truediv__(self,other):		
		if _is_constant(other): return self.__truediv__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value/other.value,
				_add_coef(_add_dim(1/other.value)*self.coef,_add_dim(-self.value/other.value**2)*other.coef))
		elif isinstance(other,np.ndarray):
			return denseAD(self.value/other,_add_dim(1./other)*self.coef)
		else:
			return denseAD(self.value/other,(1./other)*self.coef) 

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other): 	return denseAD(other/self.value,_add_dim(-other/self.value**2)*self.coef)

	def __neg__(self):		return denseAD(-self.value,-self.coef)

	# Math functions
	def __pow__(self,n): 	return denseAD(self.value**n, _add_dim(n*self.value**(n-1))*self.coef)
	def sqrt(self):		 	return self**0.5
	def log(self):			return denseAD(np.log(self.value), _add_dim(1./self.value)*self.coef)
	def exp(self):			return denseAD(np.exp(self.value), _add_dim(np.exp(self.value))*self.coef)
	def abs(self):			return denseAD(np.abs(self.value), _add_dim(np.sign(self.value))*self.coef)

	# Trigonometry
	def sin(self):			return denseAD(np.sin(self.value), _add_dim(np.cos(self.value))*self.coef)
	def cos(self):			return denseAD(np.cos(self.value), _add_dim(-np.sin(self.value))*self.coef)

	#Indexing
	@property
	def value(self): return self.view(np.ndarray)
	@property
	def size_ad(self):  return self.coef.shape[-1]

	def __getitem__(self,key):
		return denseAD(self.value[key], self.coef[key])

	def __setitem__(self,key,other):
		if isinstance(other,denseAD):
			if other.size_ad==0: return self.__setitem__(key,other.view(np.ndarray))
			elif self.size_ad==0: self.coef=np.zeros(self.coef.shape[:-1]+(other.size_ad,))
			self.value[key] = other.value
			self.coef[key] =  other.coef
		else:
			self.value[key] = other
			self.coef[key]  = 0.

	def reshape(self,shape,order='C'):
		shape2 = (shape if isinstance(shape,tuple) else (shape,))+(self.size_ad,)
		return denseAD(self.value.reshape(shape,order=order),self.coef.reshape(shape2,order=order))

	def flatten(self):	
		return self.reshape( (self.size,) )

	def broadcast_to(self,shape):
		shape2 = shape+(self.size_ad,)
		return denseAD(np.broadcast_to(self.value,shape), np.broadcast_to(self.coef,shape2) )

	@property
	def T(self):	return self if self.ndim<2 else self.transpose()
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes2 = tuple(axes) +(self.ndim,)
		return denseAD(self.value.transpose(axes),self.coef.transpose(axes2))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		out = denseAD(self.value.sum(axis,**kwargs), self.coef.sum(axis,**kwargs))
		return out

	def min(self,*args,**kwargs): return misc.min(self,*args,**kwargs)
	def max(self,*args,**kwargs): return misc.max(self,*args,**kwargs)

	def sort(self,*varargs,**kwargs):
		from . import sort
		self=sort(self,*varargs,**kwargs)


	# See https://docs.scipy.org/doc/numpy/reference/ufuncs.html
	def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):

		# Return an np.ndarray for piecewise constant functions
		if ufunc in [
		# Comparison functions
		np.greater,np.greater_equal,
		np.less,np.less_equal,
		np.equal,np.not_equal,

		# Math
		np.floor_divide,np.rint,np.sign,np.heaviside,

		# 'Floating' functions
		np.isfinite,np.isinf,np.isnan,np.isnat,
		np.signbit,np.floor,np.ceil,np.trunc
		]:
			inputs_ = (a.value if isinstance(a,denseAD) else a for a in inputs)
			return super(denseAD,self).__array_ufunc__(ufunc,method,*inputs_,**kwargs)


		if method=="__call__":

			# Reimplemented
			if ufunc==np.maximum: return misc.maximum(*inputs,**kwargs)
			if ufunc==np.minimum: return misc.minimum(*inputs,**kwargs)

			# Math functions
			if ufunc==np.sqrt: return self.sqrt()
			if ufunc==np.log: return self.log()
			if ufunc==np.exp: return self.exp()
			if ufunc==np.abs: return self.abs()

			# Trigonometry
			if ufunc==np.sin: return self.sin()
			if ufunc==np.cos: return self.cos()

			# Operators
			if ufunc==np.add: return self.add(*inputs,**kwargs)
			if ufunc==np.subtract: return self.subtract(*inputs,**kwargs)
			if ufunc==np.multiply: return self.multiply(*inputs,**kwargs)
			if ufunc==np.true_divide: return self.true_divide(*inputs,**kwargs)

		return NotImplemented


	# Numerical 
	def solve(self,shape_free=None,shape_bound=None):
		shape_free,shape_bound = misc._set_shape_free_bound(self.shape,shape_free,shape_bound)
		assert np.prod(shape_free)==self.size_ad
		v = np.moveaxis(np.reshape(self.value,(self.size_ad,)+shape_bound),0,-1)
		a = np.moveaxis(np.reshape(self.coef,(self.size_ad,)+shape_bound+(self.size_ad,)),0,-2)
		return -np.reshape(np.moveaxis(np.linalg.solve(a,v),-1,0),self.shape)

	# Static methods

	# Support for +=, -=, *=, /= 
	@staticmethod
	def add(*args,**kwargs): return misc.add(*args,**kwargs)
	@staticmethod
	def subtract(*args,**kwargs): return misc.subtract(*args,**kwargs)
	@staticmethod
	def multiply(*args,**kwargs): return misc.multiply(*args,**kwargs)
	@staticmethod
	def true_divide(*args,**kwargs): return misc.true_divide(*args,**kwargs)

	@staticmethod
	def stack(elems,axis=0):
		elems2 = tuple(denseAD(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		assert all((e.size_ad==size_ad or e.size_ad==0) for e in elems2)
		return denseAD( 
		np.stack(tuple(e.value for e in elems2), axis=axis), 
		np.stack(tuple(e.coef if e.size_ad==size_ad else np.zeros(e.shape+(size_ad,)) for e in elems2),axis=axis))

	def associate(self,singleton_axis=-1):
		from . import associate
		singleton_axis1 = singleton_axis if singleton_axis>=0 else (singleton_axis-1)
		value = associate(self.value,singleton_axis)
		coef = associate(self.coef,singleton_axis1)
		print(coef.shape)
		coef = np.moveaxis(coef,self.ndim if singleton_axis1 is None else (self.ndim-1),-1)
		print(coef.shape)
		return denseAD(value,coef)

# -------- End of class denseAD -------

# -------- Some utility functions, for internal use -------

def _add_dim(a):		return np.expand_dims(a,axis=-1)	
def _is_constant(a):	return isinstance(a,denseAD) and a.size_ad==0
def _prep_nl(s): return "\n"+s if "\n" in s else s

def _add_coef(a,b):
	if a.shape[-1]==0: return b
	elif b.shape[-1]==0: return a
	else: return a+b


# -------- Factory method -----

def identity(shape=None,shape_free=None,shape_bound=None,constant=None,shift=(0,0)):
	shape,constant = misc._set_shape_constant(shape,constant)
	shape_free,shape_bound = misc._set_shape_free_bound(shape,shape_free,shape_bound)

	ndim_elem = len(shape)-len(shape_bound)
	shape_elem = shape[:ndim_elem]
	size_elem = int(np.prod(shape_elem))
	size_ad = shift[0]+size_elem+shift[1]
	coef1 = np.full((size_elem,size_ad),0.)
	for i in range(size_elem):
		coef1[i,shift[0]+i]=1.
	coef1 = coef1.reshape(shape_elem+(1,)*len(shape_bound)+(size_ad,))
	if coef1.shape[:-1]!=constant.shape: 
		coef1 = np.broadcast_to(coef1,shape+(size_ad,))
	return denseAD(constant,coef1)
