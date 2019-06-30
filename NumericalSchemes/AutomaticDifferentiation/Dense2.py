import numpy as np
from . import Dense

class denseAD2(np.ndarray):
	"""
	A class for dense forward second order automatic differentiation
	"""

	# Construction
	# See : https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
	def __new__(cls,value,coef1=None,coef2=None):
		if isinstance(value,denseAD2):
			assert coef1 is None and coef2 is None 
			return value
		obj = np.asarray(value).view(denseAD2)
		shape1 = obj.shape+(0,)
		shape2 = obj.shape+(0,0)
		obj.coef1 = np.full(shape1,0.) if coef1  is None else coef1
		obj.coef2 = np.full(shape2,0.) if coef2  is None else coef2
		return obj

#	def __array_finalize__(self,obj): pass

	def copy(self,order='C'):
		return denseAD2(self.value.copy(order=order),self.coef1.copy(order=order),self.coef2.copy(order=order))

	# Representation 
	def __iter__(self):
		for value,coef1,coef2 in zip(self.value,self.coef1,self.coef2):
			yield denseAD2(value,coef1,coef2)

	def __str__(self):
		return "denseAD2("+str(self.value)+","+_prep_nl(str(self.coef1))+","+_prep_nl(str(self.coef2)) +")"
	def __repr__(self):
		return "denseAD2("+repr(self.value)+","+_prep_nl(repr(self.coef1))+","+_prep_nl(repr(self.coef2)) +")"

	# Operators
	def __add__(self,other):
		if isinstance(other,denseAD2):
			return denseAD2(self.value+other.value,_add_coef(self.coef1,other.coef1),_add_coef(self.coef2,other.coef2))
		else:
			return denseAD2(self.value+other, self.coef1, self.coef2)

	def __sub__(self,other):
		if isinstance(other,denseAD2):
			return denseAD2(self.value-other.value,_add_coef(self.coef1,-other.coef1),_add_coef(self.coef2,-other.coef2))
		else:
			return denseAD2(self.value-other, self.coef1, self.coef2)

	def __mul__(self,other):
		if isinstance(other,denseAD2):
			mixed = np.expand_dims(self.coef1,axis=-1)*np.expand_dims(other.coef1,axis=-2)
			return denseAD2(self.value*other.value, _add_coef(_add_dim(self.value)*other.coef1,_add_dim(other.value)*self.coef1),
				_add_coef(_add_coef(_add_dim2(self.value)*other.coef2,_add_dim2(other.value)*self.coef2),_add_coef(mixed,np.moveaxis(mixed,-2,-1))))
		elif isinstance(other,np.ndarray):
			return denseAD2(self.value*other,_add_dim(other)*self.coef1,_add_dim2(other)*self.coef2)
		else:
			return denseAD2(self.value*other,other*self.coef1,other*self.coef2)

	def __truediv__(self,other):
		if isinstance(other,denseAD2):
			return self.__mul__(other.__pow__(-1))
		elif isinstance(other,np.ndarray):
			inv = 1./other
			return denseAD2(self.value*inv,_add_dim(inv)*self.coef1,_add_dim2(inv)*self.coef2)
		else:
			inv = 1./other
			return denseAD2(self.value*inv,self.coef1*inv,self.coef2*inv)

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other):	return self.__pow__(-1).__mul__(other)


	def __neg__(self):		return denseAD2(-self.value,-self.coef1,-self.coef2)

	# Math functions
	def _math_helper(self,a,b,c): # Inputs : a=f(x), b=f'(x), c=f''(x), where x=self.value
		mixed = np.expand_dims(self.coef1,axis=-1)*np.expand_dims(self.coef1,axis=-2)
		return denseAD2(a,_add_dim(b)*self.coef1,_add_dim2(b)*self.coef2+_add_dim2(c)*mixed)
	def __pow__(self,n): 	return self._math_helper(self.value**n, n*self.value**(n-1), n*(n-1)*self.value**(n-2))
	def sqrt(self):		 	return self**0.5
	def log(self):			return self._math_helper(np.log(self.value),1./self.value,-1./self.value**2)
	def exp(self):			exp_val = np.exp(self.value); return self._math_helper(exp_val,exp_val,exp_val)
	def abs(self):			return self._math_helper(np.abs(self.value),np.sign(self.value), np.array(0.))

	# Trigonometry
	def sin(self):			sin_val = np.sin(self.value); return self._math_helper(sin_val,np.cos(self.value),-sin_val)
	def cos(self):			cos_val = np.cos(self.value); return self._math_helper(cos_val,-np.sin(self.value),-cos_val)

	#Indexing
	@property
	def value(self): return self.view(np.ndarray)
	@property
	def size_ad(self):  return self.coef1.shape[-1]

	def to_first(self): return Dense.denseAD(self.value,self.coef1)

	def __getitem__(self,key):
		return denseAD2(self.value[key], self.coef1[key], self.coef2[key])

	def __setitem__(self,key,other):
		if isinstance(other,denseAD2):
			osad = other.size_ad
			if osad==0: return self.__setitem__(key,other.view(np.ndarray))
			elif self.size_ad==0: self.coef1=np.zeros(self.coef1.shape[:-1]+(osad,)); self.coef2=np.zeros(self.coef2.shape[:-2]+(osad,osad))
			self.value[key] = other.value
			self.coef1[key] = other.coef1
			self.coef2[key] = other.coef2
		else:
			self.value[key] = other
			self.coef1[key] = 0.
			self.coef2[key] = 0.


	def reshape(self,shape,order='C'):
		shape1 = (shape if isinstance(shape,tuple) else (shape,))+(self.size_ad,)
		shape2 = (shape if isinstance(shape,tuple) else (shape,))+(self.size_ad,self.size_ad)
		return denseAD2(self.value.reshape(shape,order=order),self.coef1.reshape(shape1,order=order), self.coef2.reshape(shape2,order=order))

	def flatten(self):	
		return self.reshape( (self.size,) )

	def broadcast_to(self,shape):
		shape1 = shape+(self.size_ad,)
		shape2 = shape+(self.size_ad,self.size_ad)
		return denseAD2(np.broadcast_to(self.value,shape), np.broadcast_to(self.coef1,shape1), np.broadcast_to(self.coef2,shape2))

	@property
	def T(self):	return self if self.ndim<2 else self.transpose()
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes1 = tuple(axes) +(self.ndim,)
		axes2 = tuple(axes) +(self.ndim,self.ndim+1)
		return denseAD2(self.value.transpose(axes),self.coef1.transpose(axes1),self.coef2.transpose(axes2))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		out = denseAD2(self.value.sum(axis,**kwargs), self.coef1.sum(axis,**kwargs), self.coef2.sum(axis,**kwargs))
		return out

	def min(self,axis=0,keepdims=False,out=None):
		ai = np.expand_dims(np.argmin(self.value, axis=axis), axis=axis)
		out = np.take_along_axis(self,ai,axis=axis)
		if not keepdims: out = out.reshape(self.shape[:axis]+self.shape[axis+1:])
		return out

	def max(self,axis=0,keepdims=False,out=None):
		ai = np.expand_dims(np.argmax(self.value, axis=axis), axis=axis)
		out = np.take_along_axis(self,ai,axis=axis)
		if not keepdims: out = out.reshape(self.shape[:axis]+self.shape[axis+1:])
		return out

	def sort(self,*varargs,**kwargs):
		from . import sort
		self=sort(self,*varargs,**kwargs)


	# See https://docs.scipy.org/doc/numpy/reference/ufuncs.html
	def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):

#		if ufunc!=np.maximum:
#			print(self)
#			return NotImplemented
		# Return an np.ndarray for piecewise constant functions
		if ufunc in [
		# Comparison functions
		np.greater,np.greater_equal,
		np.less,np.less_equal,
		np.equal,np.not_equal,

		# Math
		np.floor_divide,np.rint,np.sign,np.heaviside,

		# Floating functions
		np.isfinite,np.isinf,np.isnan,np.isnat,
		np.signbit,np.floor,np.ceil,np.trunc
		]:
			inputs_ = (a.value if isinstance(a,denseAD2) else a for a in inputs)
			return super(denseAD2,self).__array_ufunc__(ufunc,method,*inputs_,**kwargs)


		if method=="__call__":

			# Reimplemented
			if ufunc==np.maximum: return maximum(*inputs,**kwargs)
			if ufunc==np.minimum: return minimum(*inputs,**kwargs)

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
	
	# Static methods

	# Support for +=, -=, *=, /=
	@staticmethod
	def add(a,b,out=None,where=True): 
		if out is None: return a+b #if isinstance(a,denseAD2) else b+a; 
		else: result=_tuple_first(out); result[where]=a[where]+b[where]; return result

	@staticmethod
	def subtract(a,b,out=None,where=True):
		if out is None: return a-b #if isinstance(a,denseAD2) else b.__rsub__(a); 
		else: result=_tuple_first(out); result[where]=a[where]-b[where]; return result

	@staticmethod
	def multiply(a,b,out=None,where=True): 
		if out is None: return a*b #if isinstance(a,denseAD2) else b*a; 
		else: result=_tuple_first(out); result[where]=a[where]*b[where]; return result

	@staticmethod
	def true_divide(a,b,out=None,where=True): 
		if out is None: return a/b #if isinstance(a,denseAD2) else b.__rtruediv__(a); 
		else: result=_tuple_first(out); result[where]=a[where]/b[where]; return result

	@staticmethod
	def stack(elems,axis=0):
		elems2 = tuple(denseAD2(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		assert all((e.size_ad==size_ad or e.size_ad==0) for e in elems2)
		return denseAD( 
		np.stack(tuple(e.value for e in elems2), axis=axis), 
		np.stack(tuple(e.coef1 if e.size_ad==size_ad else np.zeros(e.shape+(size_ad,)) for e in elems2),axis=axis),
		np.stack(tuple(e.coef2 if e.size_ad==size_ad else np.zeros(e.shape+(size_ad,size_ad)) for e in elems2),axis=axis))

# -------- End of class denseAD2 -------

# -------- Some utility functions, for internal use -------

def _concatenate(a,b): 	return np.concatenate((a,b),axis=-1)
def _add_dim(a):		return np.expand_dims(a,axis=-1)	
def _add_dim2(a):		return _add_dim(_add_dim(a))
def _tuple_first(a): return a[0] if isinstance(a,tuple) else a
def _prep_nl(s): return "\n"+s if "\n" in s else s

def _add_coef(a,b):
	if a.shape[-1]==0: return b
	elif b.shape[-1]==0: return a
	else: return a+b

# -------- Factory method -----

def identity(*args,**kwargs):
	arr = Dense.identity(*args,**kwargs)
	return denseAD2(arr.value,arr.coef,np.zeros(arr.shape+(arr.size_ad,arr.size_ad)))

# ----- Operators -----

#def add(a,other,out=None):	out=self+other; return out

# ----- Various functions, intended to be numpy-compatible ------


def maximum(a,b): 	
	from . import where
	return where(a>b,a,b)
def minimum(a,b): 	
	from . import where
	return where(a<b,a,b)


