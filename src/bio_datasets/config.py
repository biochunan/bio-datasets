import importlib

FOLDCOMP_AVAILABLE = importlib.util.find_spec("foldcomp") is not None
