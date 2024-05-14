import inspect

from .adalam import AdalamMatcher
from .kornia_matcher import KorniaMatcher
from .lightglue import LightGlueMatcher
from .loftr import LOFTRMatcher
from .matcher_base import DetectorFreeMatcherBase, FeaturesDict, MatcherBase
from .roma import RomaMatcher
from .se2loftr import SE2LOFTRMatcher
from .superglue import SuperGlueMatcher


# Dynamic loading
def matcher_loader(root, model):
    """
    Load a matcher class from a specified module.

    Args:
        root (module): The root module containing the specified matcher module.
        model (str): The name of the matcher module to load.

    Returns:
        type: The matcher class.
    """
    module_path = f"{root.__name__}.{model}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    # classes = [c for c in classes if issubclass(c[1], MatcherBase)]
    classes = [c for c in classes if issubclass(c[1], MatcherBase) or issubclass(c[1], DetectorFreeMatcherBase)]
    assert len(classes) == 1, classes
    return classes[0][1]


# For dynamic loading
# def get_matcher(matcher):
#     mod = __import__(f"{__name__}.{matcher}", fromlist=[""])
#     return getattr(mod, "Model")
