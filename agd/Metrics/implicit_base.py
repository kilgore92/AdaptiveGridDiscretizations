import numpy as np
from .. import AutomaticDifferentiation as ad
from .. import LinearParallel as lp
from .base import Base


class ImplicitBase(object):
	"""
	Base class for a metric defined implicitly, 
	in terms of a level set function for the unit ball
	of the dual metric, and of a rotation.
	"""

	def norm(self,v):
		pass