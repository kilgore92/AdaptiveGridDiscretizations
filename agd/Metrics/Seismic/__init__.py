from .implicit_base import ImplicitBase
from .reduced import Reduced
from .hooke 	import Hooke

def reload_submodules():
	from importlib import reload
	import sys
	seismic = sys.modules['agd.Metrics.Seismic']

	global ImplicitBase
	seismic.implicit_base = reload(seismic.implicit_base)
	ImplicitBase = implicit_base.ImplicitBase

	global Reduced
	seismic.reduced = reload(seismic.reduced)
	Reduced = seismic.Reduced

	global Hooke
	seismic.hooke = reload(seismic.hooke)
	Hooke = hooke.Hooke






