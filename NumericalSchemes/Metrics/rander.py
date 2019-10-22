from .base import Base
from .riemann import Riemann
from .. import LinearParallel as lp

class Rander(Base):
	"""
A Rander norm takes the form F(x) = sqrt(<x,m.x>) + <w,x>.
Inputs : 
- m : Symmetric positive definite matrix.
- w : Vector, obeying <w,m^(-1).w> < 1
"""
	def __init__(self,m,w):
		self.m=m
		self.w=w

	def norm(self,v):
		v,m,w = misc.common_field((v,self.m,w),(1,2,1))
		return np.sqrt(lp.dot_VAV(v,m,v))+lp.dot_VV(v,w)

	def is_definite(self):
		return np.logical_and(Riemann(self.m).is_definite(),
			lp.dot_VV(lp.solve_AV(self.m,self.w),self.w) <1.)


	def dual(self):
		"""
This function returns the dual 
to a Rander norm, which turns out to have a similar algebraic form.
The dual norm is defined as 
	F'(x) = sup{ <x,y>; F(y)<=1 }
"""
		s = lp.inverse(m-lp.outer_self(w))
		omega = lp.dot_AV(s,w)
		return Rander((1+lp.dot_VV(w,omega))*s, omega)

	@property
	def ndim(self): return len(self.m)
	
	def inv_transform(self,a):
		return Rander(Riemann(self.m).inv_transform(a),lp.dot_VA(w,a))

	def flatten(self):
		return np.concatenate((misc.flatten_symmetric_matrix(m),w),axis=0)
	
	@classmethod
	def expand(cls,arr):
		m = misc.expand_symmetric_matrix(arr)
		d=len(m)
		d_sym = (d*(d+1))//2
		assert(len(arr)==d_sym+d)
		w=arr[d_sym:]
		return cls(m,w)

	@classmethod
	def from_Zermelo(cls,metric,drift):
		"""
Zermelo's navigation problem consists in computing a minimal path for 
whose velocity is unit w.r.t. a Riemannian metric, and which is subject 
to a drift. The accessible velocities take the form 
	x+drift where <x,m.x> <= 1
This function reformulates it as a shortest path problem 
in a Rander manifold.
Inputs : 
- metric : Symmetric positive definite matrix (Riemannian metric)
- drift : Vector field, obeying <drift,metric.drift> < 1 (Drift)
Outputs : 
- the Rander metric.
"""
		return cls(lp.inverse(metric),-drift).dual()

	def to_Zermelo(self):
		"""
Input : Parameters of a Rander metric.
Output : Parameters of the corresponding Zermelo problem, of motion on a 
Riemannian manifold with a drift.
"""
		self_dual = self.dual()
		return lp.inverse(self_dual.m), -self_dual.w

	def to_Varadhan(eps=1):
		"""
The Rander eikonal equation can be reformulated in an (approximate)
linear form, using a logarithmic transformation
	u + 2 eps <omega,grad u> - eps**2 Tr(D hess u).
Then -eps log(u) solves the Rander eikonal equation, 
up to a small additional diffusive term.
Inputs : 
- m and w, parameters of the Rander metric
- eps (optionnal), relaxation parameter
Outputs : 
- D and 2*omega, parameters of the linear PDE. 
 (D*eps**2 and 2*omega*eps if eps is specified)
"""
		return self.Varadhan_from_Zermelo(self.to_Zermelo(),eps)

	@staticmethod
	def Varadhan_from_Zermelo(metric,drift,eps=1):
		"""
Zermelo's navigation problem can be turned into a Rander shortest path problem,
which itself can be (approximately) expressed in linear form using the logarithmic
transformation. This function composes the above two steps.
"""
		return eps**2*(lp.inverse(metric)-lp.outer_self(drift)), 2.*eps*drift
