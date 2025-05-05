# dynamic versioning
from importlib.metadata import version, PackageNotFoundError
from sen2sr.utils import predict_large
from sen2sr.xai.lam import lam

try:
    __version__ = version("sen2sr")
except PackageNotFoundError:    
    __version__ = "unknown"


try:
    import torch    
except ImportError:
    raise("SEN2SR: PyTorch is not available. Install it to use this package.")
    

__all__ = [
    "__version__",
    "predict_large",
    "lam"
]