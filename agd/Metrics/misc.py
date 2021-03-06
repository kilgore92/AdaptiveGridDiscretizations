from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd

def flatten_symmetric_matrix(m):
	"""
	Input : a square (symmetric) matrix.
	Output : a vector containing the lower triangular entries
	"""
	d=m.shape[0]
	assert(d==m.shape[1])
	return ad.array([ m[i,j] for i in range(d) for j in range(i+1)])

def expand_symmetric_matrix(arr,d=None,extra_length=False):
	if d is None:
		d=0
		while (d*(d+1))//2 < len(arr):
			d+=1
	assert(extra_length or len(arr)==(d*(d+1))//2)
	
	def index(i,j):
		i,j = min(i,j),max(i,j)
		return (i*(i+1))//2+j
	return ad.array([ [ arr[index(i,j)] for i in range(d)] for j in range(d) ])



