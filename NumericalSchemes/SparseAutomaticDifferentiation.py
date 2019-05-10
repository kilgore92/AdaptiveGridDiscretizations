import numpy as np

class spAD:
	"""
	A class for sparse forward automatic differentiation
	"""
	def __init__(self,value,coef,index):
		self.value = value
		self.coef = _add_dim(coef) if coef.shape==value.shape else coef 
		self.index = _add_dim(index) if index.shape==value.shape else index 

	def __iter__(self):
		for value,coef,index in zip(self.value,self.coef,self.index):
			yield spAD(value,coef,index)

	@staticmethod
	def constant(value):
		return spAD(value,np.full(value.shape+(0,),0.),np.full(value.shape+(0,),0) )

	@staticmethod
	def identity(value):
		return spAD(value,np.full(value.shape+(1,),1.),np.arange(value.size).reshape(value.shape+(1,)))

	def __str__(self):
		return "spAD( "+str(self.value)+ ")"
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
	def __rsub__(self,other): 		return __sub__(constant(other),self)
	def __rtruediv__(self,other): 	return __truediv__(constant(other),self)

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

	def take(self, indices,**kwargs):
		return spAD(np.take(self.value,indices,**kwargs), np.take(self.coef,indices,**kwargs), np.take(self.index,indices,**kwargs))

	def reshape(self,shape):
		shape2 = shape+(self.coef.shape[-1],)
		return spAD(self.value.reshape(shape),self.coef.reshape(shape2), self.indices.reshape(shape2))

	def flatten(self,shape):	
		return self.reshape( (self.size,) )

	def triplets(self):
		coef = self.coef.flatten()
		row = np.broadcast_to(_add_dim(np.arange(self.shape).reshape(self.shape)), self.indices.shape).flatten()
		column = self.indices.flatten()

		pos=coef!=0
		return (coef[pos],(row[pos],column[pos]))

# -------- Some utility functions, for internal use -------

def _concatenate(a,b): return np.concatenate((a,b),axis=-1)
def _add_dim(a):	return np.expand_dims(a,axis=-1)	
def _get_value(a): return a.value if isinstance(a,spAD) else a
def _pad_last(a,pad_total):
		return np.pad(a, pad_width=((0,0),)*(a.ndim-1)+((0,pad_total-a.shape[-1]),), mode='constant', constant_values=0)


# -------- End of class spAD -------

def replace_at(a,mask,b): 
	if isinstance(a,spAD): return a.replace_at(mask,b) 
	elif isinstance(b,spAD): return b.replace_at(np.logical_not(mask),a) 
	else: result=np.copy(a); result[mask]=b; return result

def maximum(a,b): return replace_at(a,a<b,b)
def minimum(a,b): return replace_at(a,a>b,b)



