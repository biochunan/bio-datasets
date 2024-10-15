import inspect
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES, _hash_python_lines
from .features import *
from .protein import Protein
from .packaged_modules.structurefolder import structurefolder


_PACKAGED_BIO_MODULES = {
    "structurefolder": (structurefolder.__name__, _hash_python_lines(inspect.getsource(structurefolder).splitlines()))
}

_PACKAGED_DATASETS_MODULES.update(_PACKAGED_BIO_MODULES)