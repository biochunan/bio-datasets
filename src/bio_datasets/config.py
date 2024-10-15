import importlib

FOLDCOMP_AVAILABLE = importlib.util.find_spec("foldcomp") is not None
FASTPDB_AVAILABLE = importlib.util.find_spec("fastpdb") is not None
