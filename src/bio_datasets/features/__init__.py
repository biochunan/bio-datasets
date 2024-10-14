__all__ = ["AtomArray", "Structure"]

from datasets.features.features import register_feature

from .atom_array import AtomArray, Structure

register_feature(Structure, "Structure")
register_feature(AtomArray, "AtomArray")
