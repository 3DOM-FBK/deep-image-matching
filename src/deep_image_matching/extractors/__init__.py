import inspect

from .alike import AlikeExtractor
from .aliked import AlikedExtractor
from .dedode import DeDoDeExtractor
from .disk import DiskExtractor
from .extractor_base import ExtractorBase, FeaturesDict
from .keynetaffnethardnet import KeyNetExtractor
from .no_extractor import NoExtractor
from .orb import ORBExtractor
from .sift import SIFTExtractor
from .superpoint import SuperPointExtractor
from .xfeat import XFeatExtractor


def extractor_loader(root, model):
    """
    Load and return the specified extractor class from the given root module.

    Args:
        root (module): The root module where the extractor module is located.
        model (str): The name of the extractor module.

    Returns:
        class: The specified extractor class.

    Raises:
        AssertionError: If no or multiple extractor classes are found.

    """
    module_path = f"{root.__name__}.{model}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], ExtractorBase)]
    assert len(classes) == 1, classes
    return classes[0][1]

