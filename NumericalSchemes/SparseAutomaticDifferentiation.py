import numpy as np

class spAD:
	"""
	A class for sparse forward automatic differentiation
	"""

	# Construction
	def __init__(self,value,coef=None,index=None):
		self.value = value
		self.coef  = np.full(value.shape+(0,),0.) if coef is None else coef
		self.index = np.full(value.shape+(0,),0) if index is None else index 

#	@staticmethod
#	def constant(value):
#		return spAD(value,np.full(value.shape+(0,),0.),np.full(value.shape+(0,),0) )

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
			return spAD(self.value*other.value, 
				_concatenate(_add_dim(other.value)*self.coef,_add_dim(self.value)*other.coef), 
				_concatenate(self.index,other.index))
		elif isinstance(other,np.ndarray):
			return spAD(self.value*other, _add_dim(other)*self.coef, self.index)
		else:
			return spAD(self.value*other,other*self.coef,self.index)

	def __truediv__(self,other):
		if isinstance(other,spAD):
			return spAD(self.value/other.value,
				_concatenate(self.coef*_add_dim(1/other.value),other.coef*_add_dim(-self.value/other.value**2)),
				_concatenate(self.index,other.index))
		elif isinstance(other,np.ndarray):
			return spAD(self.value/other,self.coef*_add_dim(1/other),self.index)
		else:
			return spAD(self.value/other,self.coef/other,self.index)

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other): 	return self.__truediv__(self.constant(other),self)

	def __neg__(self):		return spAD(-self.value,-self.coef,self.index)
	def __pow__(self,n): 	return spAD(self.value**n, _add_dim(n*self.value**(n-1))*self.coef,self.index)
	def sqrt(self):		 	return self**0.5
	def log(self):			return spAD(np.log(self.value), self.coef*_add_dim(1./self.value), self.index)
	def exp(self):			return spAD(np.exp(self.value), self.coef*_add_dim(np.exp(self.value)), self.index)
	def sin(self):			return spAD(np.sin(self.value), self.coef*_add_dim(np.cos(self.value)), self.index)
	def cos(self):			return spAD(np.cos(self.value), self.coef*_add_dim(-np.sin(self.value)), self.index)

	def __lt__(self,other): return self.value <  _get_value(other)
	def __le__(self,other): return self.value <= _get_value(other)
	def __eq__(self,other): return self.value == _get_value(other)
	def __ne__(self,other): return self.value != _get_value(other)
	def __gt__(self,other): return self.value >  _get_value(other)
	def __ge__(self,other): return self.value >= _get_value(other)

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
	def shape(self):	return self.value.shape
	@property
	def size(self):		return self.value.size
	@property
	def size_ad(self):  return self.coef.shape[-1]
	

	def __getitem__(self,key):
		return spAD(self.value[key], self.coef[key], self.index[key])

	def __setitem__(self,key,other):
		self.value[key] = other.value
		if isinstance(other,spAD):
			pad_size = max(self.coef.shape[-1],other.coef.shape[-1])
			if pad_size>self.coef.shape[-1]:
				self.coef = _pad_last(self.coef,pad_size)
				self.index = _pad_last(self.index,pad_size)
			self.coef[key] = _pad_last(other.coef,pad_size)
			self.index[key] = _pad_last(other.index,pad_size)
		else:
			self.coef[key] = 0.
			self.index[key] = 0

	def reshape(self,shape):
		shape2 = shape+(self.coef.shape[-1],)
		return spAD(self.value.reshape(shape),self.coef.reshape(shape2), self.index.reshape(shape2))

	def flatten(self):	
		return self.reshape( (self.size,) )

	def triplets(self):
		coef = self.coef.flatten()
		row = np.broadcast_to(_add_dim(np.arange(self.size).reshape(self.shape)), self.index.shape).flatten()
		column = self.index.flatten()

		pos=coef!=0
		return (coef[pos],(row[pos],column[pos]))

	def sum(self,axis=0):
		value = self.value.sum(axis)
		shape = value.shape +(self.size_ad * self.shape[axis],)
		coef = np.moveaxis(self.coef, axis,-1).reshape(shape)
		index = np.moveaxis(self.index, axis,-1).reshape(shape)
		return spAD(value,coef,index)

# -------- End of class spAD -------

# -------- Some utility functions, for internal use -------

def _concatenate(a,b): return np.concatenate((a,b),axis=-1)
def _add_dim(a):	return np.expand_dims(a,axis=-1)	
def _get_value(a): return a.value if isinstance(a,spAD) else a
def _pad_last(a,pad_total):
		return np.pad(a, pad_width=((0,0),)*(a.ndim-1)+((0,pad_total-a.shape[-1]),), mode='constant', constant_values=0)

# -------- Various functions, including factory methods -----

def replace_at(a,mask,b): 
	if isinstance(a,spAD): return a.replace_at(mask,b) 
	elif isinstance(b,spAD): return b.replace_at(np.logical_not(mask),a) 
	else: result=np.copy(a); result[mask]=b[mask] if isinstance(b,np.ndarray) else b; return result

def maximum(a,b): return replace_at(a,a<b,b)
def minimum(a,b): return replace_at(a,a>b,b)
def abs(a,b): return replace_at(a,a<0,-a)


def identity(shape):
	return spAD(np.full(shape,0.),np.full(shape+(1,),1.),np.arange(np.prod(shape)).reshape(shape+(1,)))

def stack(elems,axis=0):
	return spAD( 
		np.stack((e.value for e in elems),axis=axis), 
		np.stack((e.coef  for e in elems),axis=axis),
		np.stack((e.index for e in elems),axis=axis))

def cast_left_operand(u,v):
	"""Returns u, or a cast of u if necessary to ensure compatibility for u+v, u*v, u-v, u/v, etc"""
	return spAD(u) if (type(u)==np.ndarray and type(v)==spAD) else u



