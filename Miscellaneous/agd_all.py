
from agd import AutomaticDifferentiation
from agd import Domain
from agd import FiniteDifferences as fd
from agd import HFMUtils
from agd import LinearParallel as lp
from agd.Metrics import Seismic
from agd import Metrics
from agd import Plotting
from agd import Selling

def agd_subdirs(root_dir=""):
    import os
    return [os.path.join(root_dir,"agd",sub_dir) for sub_dir in 
    ["HFMUtils","AutomaticDifferentiation","Metrics","Metrics/Seismic"] ]

def rreload(module, paths=None, mdict=None, base_module=None, blacklist=None, reloaded_modules=None):
    """
    Recursively reload modules.
    https://stackoverflow.com/a/58201660/12508258
    """
    from importlib import reload
    from types import ModuleType
    import os, sys
    if paths is None:
        paths = [""]
    if mdict is None:
        mdict = {}
    if module not in mdict:
        # modules reloaded from this module
        mdict[module] = []
    if base_module is None:
        base_module = module
    if blacklist is None:
        blacklist = ["importlib", "typing"]
    if reloaded_modules is None:
        reloaded_modules = []
    reload(module)
    reloaded_modules.append(module.__name__)
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if type(attribute) is ModuleType and attribute.__name__ not in blacklist:
            if attribute not in mdict[module]:
                if attribute.__name__ not in sys.builtin_module_names:
                    if os.path.dirname(attribute.__file__) in paths:
                        mdict[module].append(attribute)
                        reloaded_modules = rreload(attribute, paths, mdict, base_module, blacklist, reloaded_modules)
        elif callable(attribute) and attribute.__module__ not in blacklist:
            if attribute.__module__ not in sys.builtin_module_names and f"_{attribute.__module__}" not in sys.builtin_module_names:
                if sys.modules[attribute.__module__] != base_module:
                    if sys.modules[attribute.__module__] not in mdict:
                        mdict[sys.modules[attribute.__module__]] = [attribute]
                        reloaded_modules = rreload(sys.modules[attribute.__module__], paths, mdict, base_module, blacklist, reloaded_modules)
    reload(module)
    return reloaded_modules