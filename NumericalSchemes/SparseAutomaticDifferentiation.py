import numpy as np

class spAD(np.ndarray):
	"""
	A class for sparse forward automatic differentiation
	"""

	# Construction
	# See : https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
	def __new__(cls,value,coef=None,index=None):
		if isinstance(value,spAD):
			assert coef is None and index is None
			return value
		obj = np.asarray(value).view(spAD)
		shape2 = obj.shape+(0,)
		obj.coef  = np.full(shape2,0.) if coef  is None else coef
		obj.index = np.full(shape2,0)  if index is None else index
		return obj

#	def __array_finalize__(self,obj): pass

	def copy(self,order='C'):
		return spAD(self.value.copy(order=order),self.coef.copy(order=order),self.index.copy(order=order))

	# Representation 
	def __iter__(self):
		for value,coef,index in zip(self.value,self.coef,self.index):
			yield spAD(value,coef,index)

	def __str__(self):
		return "spAD "+str((self.value,self.coef,self.index))
	def __repr__(self):
		return "spAD"+repr((self.value,self.coef,self.index))	

	# Operators
	def __add__(self,other):
		if isinstance(other,spAD):
			return spAD(self.value+other.value, _concatenate(self.coef,other.coef), _concatenate(self.index,other.index))
		else:
			return spAD(self.value+other, self.coef, self.index)

	def __sub__(self,other):
		if isinstance(other,spAD):
			return spAD(self.value-other.value, _concatenate(self.coef,-other.coef), _concatenate(self.index,other.index))
		else:
			return spAD(self.value-other, self.coef, self.index)

	def __mul__(self,other):
		if isinstance(other,spAD):
			value = self.value*other.value
			coef1,coef2 = _add_dim(other.value)*self.coef,_add_dim(self.value)*other.coef
			index1,index2 = np.broadcast_to(self.index,coef1.shape),np.broadcast_to(other.index,coef2.shape)
			return spAD(value,_concatenate(coef1,coef2),_concatenate(index1,index2))
		elif isinstance(other,np.ndarray):
			value = self.value*other
			coef = _add_dim(other)*self.coef
			index = np.broadcast_to(self.index,coef.shape)
			return spAD(value,coef,index)
		else:
			return spAD(self.value*other,other*self.coef,self.index)

	def __truediv__(self,other):
		if isinstance(other,spAD):
			return spAD(self.value/other.value,
				_concatenate(self.coef*_add_dim(1/other.value),other.coef*_add_dim(-self.value/other.value**2)),
				_concatenate(self.index,other.index))
		elif isinstance(other,np.ndarray):
			return spAD(self.value/other,self.coef*_add_dim(1./other),self.index)
		else:
			return spAD(self.value/other,self.coef/other,self.index)

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other): 	spAD(other/self.value,self.coef*_add_dim(-other/self.value**2),self.index)

	def __neg__(self):		return spAD(-self.value,-self.coef,self.index)

	# Math functions
	def __pow__(self,n): 	return spAD(self.value**n, _add_dim(n*self.value**(n-1))*self.coef,self.index)
	def sqrt(self):		 	return self**0.5
	def log(self):			return spAD(np.log(self.value), self.coef*_add_dim(1./self.value), self.index)
	def exp(self):			return spAD(np.exp(self.value), self.coef*_add_dim(np.exp(self.value)), self.index)
	def abs(self):			return spAD(np.abs(self.value), self.coef*_add_dim(np.sign(self.value)), self.index)

	# Trigonometry
	def sin(self):			return spAD(np.sin(self.value), self.coef*_add_dim(np.cos(self.value)), self.index)
	def cos(self):			return spAD(np.cos(self.value), self.coef*_add_dim(-np.sin(self.value)), self.index)

	#Indexing

	def replace_at(self,mask,other):
		if isinstance(other,spAD):
			value = np.copy(self.value)
			value[mask] = other.value[mask]

			pad_size = max(self.coef.shape[-1],other.coef.shape[-1])
			coef = _pad_last(self.coef,pad_size)
			coef[mask] = _pad_last(other.coef,pad_size)[mask]

			index = _pad_last(self.index,pad_size)
			index[mask] = _pad_last(other.index,pad_size)[mask]

			return spAD(value,coef,index)
		else:
			value,coef,index = np.copy(self.value), np.copy(self.coef), np.copy(self.index)
			value[mask]=other[mask] if isinstance(other,np.ndarray) else other
			coef[mask]=0.
			index[mask]=0
			return spAD(value,coef,index)

	@property
	def value(self): return self.view(np.ndarray)
	@property
	def size_ad(self):  return self.coef.shape[-1]
	
	def __getitem__(self,key):
		return spAD(self.value[key], self.coef[key], self.index[key])

	def __setitem__(self,key,other):
		if isinstance(other,spAD):
			self.value[key] = other.value
			pad_size = max(self.coef.shape[-1],other.coef.shape[-1])
			if pad_size>self.coef.shape[-1]:
				self.coef = _pad_last(self.coef,pad_size)
				self.index = _pad_last(self.index,pad_size)
			self.coef[key] = _pad_last(other.coef,pad_size)
			self.index[key] = _pad_last(other.index,pad_size)
		else:
			self.value[key] = other
			self.coef[key]  = 0.
			self.index[key] = 0

	def reshape(self,shape):
		shape2 = shape+(self.size_ad,)
		return spAD(self.value.reshape(shape),self.coef.reshape(shape2), self.index.reshape(shape2))

	def flatten(self):	
		return self.reshape( (self.size,) )

	def triplets(self):
		coef = self.coef.flatten()
		row = np.broadcast_to(_add_dim(np.arange(self.size).reshape(self.shape)), self.index.shape).flatten()
		column = self.index.flatten()

		pos=coef!=0
		return (coef[pos],(row[pos],column[pos]))

	# Reductions

	def sum(self,axis=0,out=None,**kwargs):
		value = self.value.sum(axis,**kwargs)
		shape = value.shape +(self.size_ad * self.shape[axis],)
		coef = np.moveaxis(self.coef, axis,-1).reshape(shape)
		index = np.moveaxis(self.index, axis,-1).reshape(shape)
		out = spAD(value,coef,index)
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

		# Floating functions
		np.isfinite,np.isinf,np.isnan,np.isnat,
		np.signbit,np.floor,np.ceil,np.trunc
		]:
			inputs_ = (a.value if isinstance(a,spAD) else a for a in inputs)
			return super(spAD,self).__array_ufunc__(ufunc,method,*inputs_,**kwargs)

		assert method=="__call__"

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

	# Support for +=, -=, *=, /=
	@staticmethod
	def add(a,b,out=None,where=True): 
		if out is None: return a+b; 
		else: result=_tuple_first(out); result[where]=a[where]+b[where]; return result

	@staticmethod
	def subtract(a,b,out=None,where=True): 
		if out is None: return a-b; 
		else: result=_tuple_first(out); result[where]=a[where]-b[where]; return result

	@staticmethod
	def multiply(a,b,out=None,where=True): 
		if out is None: return a*b; 
		else: result=_tuple_first(out); result[where]=a[where]*b[where]; return result

	@staticmethod
	def true_divide(a,b,out=None,where=True): 
		if out is None: return a/b; 
		else: result=_tuple_first(out); result[where]=a[where]/b[where]; return result


# -------- End of class spAD -------

# -------- Some utility functions, for internal use -------

def _concatenate(a,b): 	return np.concatenate((a,b),axis=-1)
def _add_dim(a):		return np.expand_dims(a,axis=-1)	
def _get_value(a): 		return a.value if isinstance(a,spAD) else a
def _pad_last(a,pad_total):
		return np.pad(a, pad_width=((0,0),)*(a.ndim-1)+((0,pad_total-a.shape[-1]),), mode='constant', constant_values=0)
def _tuple_first(a): return a[0] if isinstance(a,tuple) else a

# -------- Various functions, including factory methods -----

def identity(shape=None,constant=None):
	if shape is None:
		if constant is None:
			raise ValueError("identity error : shape or constant term must be specified")
		shape = constant.shape
	if constant is None:
		constant = np.full(shape,0.)
	shape2 = shape+(1,)
	return spAD(constant,np.full(shape2,1.),np.arange(np.prod(shape)).reshape(shape2))


#def simplify_ad(array): #TODO
#	if not isinstance(array,spAD): return array

# ----- Operators -----

def add(a,other,out=None):	out=self+other; return out


# ----- Various functions, intended to be numpy-compatible ------

def where(mask,a,b): 
	if isinstance(b,spAD): return b.replace_at(mask,a) 
	elif isinstance(a,spAD): return a.replace_at(np.logical_not(mask),b) 
	else: return np.where(mask,a,b)

def maximum(a,b): 	return where(a>b,a,b)
def minimum(a,b): 	return where(a<b,a,b)

def stack(elems,axis=0):
	if any(isinstance(e,spAD) for e in elems):
		elems2 = tuple(spAD(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		return spAD( 
		np.stack(tuple(e.value for e in elems2), axis=axis), 
		np.stack(tuple(_pad_last(e.coef,size_ad)  for e in elems2),axis=axis),
		np.stack(tuple(_pad_last(e.index,size_ad) for e in elems2),axis=axis))
	else:
		return np.stack(elems,axis=axis) 


def broadcast_to(array,shape):
	if not isinstance(array,spAD):
		return np.broadcast_to(array,shape)
	shape2 = shape+(array.size_ad,)
	return spAD(np.broadcast_to(array.value,shape), np.broadcast_to(array.coef,shape2), np.broadcast_to(array.index,shape2))

def sort(array,axis=-1,*varargs,**kwargs):
	if not isinstance(array,spAD):
		return np.sort(array,axis=axis,*varargs,**kwargs)
	ai = np.argsort(array.value,axis=axis,*varargs,**kwargs)
	return np.take_along_axis(array,ai,axis=axis)
