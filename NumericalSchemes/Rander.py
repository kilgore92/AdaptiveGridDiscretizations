from NumericalSchemes import LinearParallel as lp
from NumericalSchemes import FiniteDifferences as fd
import numpy as np

# This file implement conversion utilities between the common forms of a Rander geometry.

def Norm(m,w):
	"""
A Rander norm takes the form F(x) = sqrt(<x,m.x>) + <w,x>.
Inputs : 
- m : Symmetric positive definite matrix.
- w : Vector, obeying <w,m^(-1).w> < 1
Output : 
- Rander norm object
"""
	def Norm(x,as_field=True):
		nonlocal m,w
		if as_field and x.ndim > 1:
			m,w = (fd.as_field(e,x.shape[1:],False) for e in (m,w))
		return np.sqrt(lp.dot_VAV(x,m,x)) + lp.dot_VV(w,x)
	return Norm

def Dual(m,w):
	"""
This function returns the parameters for the dual norm
to a Rander norm, which turns out to have a similar algebraic form.
The dual norm is defined as 
	F'(x) = sup{ <x,y>; F(y)<=1 }
Inputs :
- m and w, Rander parameters 
Outputs :
- (m',w'), parameters of the dual metric.
"""
	s = lp.inverse(m-lp.outer_self(w))
	omega = lp.dot_AV(s,w)
	return (1+lp.dot_VV(w,omega))*s, omega

def FromZermelo(metric,drift):
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
- (m,w), parameters of the Rander metric.
"""
	return Dual(lp.inverse(metric),-drift)

def ToZermelo(m,w):
	"""
Input : Parameters of a Rander metric.
Output : Parameters of the corresponding Zermelo problem, of motion on a 
Riemannian manifold with a drift.
"""
	mp,wp = Dual(m,w)
	return lp.inverse(mp), -wp


def ToVaradhan(m,w,eps=1):
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
	return VaradhanFromZermelo(*ToZermelo(m,w),eps)

def VaradhanFromZermelo(metric,drift,eps=1):
	"""
Zermelo's navigation problem can be turned into a Rander shortest path problem,
which itself can be (approximately) expressed in linear form using the logarithmic
transformation. This function composes the above two steps.
"""
	return eps**2*(lp.inverse(metric)-lp.outer_self(drift)), 2.*eps*drift
