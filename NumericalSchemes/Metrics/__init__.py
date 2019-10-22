from .isotropic import Isotropic
from .riemann 	import Riemann
from .rander 	import Rander
from .asym_quad import AsymQuad
from .hooke 	import Hooke

def reload_submodules():
	from importlib import reload
	import sys
	metrics = sys.modules['NumericalSchemes.Metrics']

	global Isotropic
	metrics.isotropic = reload(metrics.isotropic)
	Isotropic = isotropic.Isotropic
	
	global Riemann
	metrics.riemann = reload(metrics.riemann)
	Riemann = riemann.Riemann

	global Rander
	metrics.rander = reload(metrics.rander)
	Rander = rander.Rander

	global AsymQuad
	metrics.asym_quad = reload(metrics.asym_quad)
	AsymQuad = asym_quad.AsymQuad

	global Hooke
	metrics.hooke = reload(metrics.hooke)
	Hooke = hooke.Hooke