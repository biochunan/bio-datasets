__all__ = [
    "AtomArrayFeature",
    "StructureFeature",
    "ProteinAtomArrayFeature",
    "ProteinStructureFeature",
]

from datasets.features.features import register_feature

from .atom_array import (
    AtomArrayFeature,
    ProteinAtomArrayFeature,
    ProteinStructureFeature,
    StructureFeature,
)

register_feature(StructureFeature, "Structure")
register_feature(AtomArrayFeature, "AtomArray")
register_feature(ProteinAtomArrayFeature, "Protein")
register_feature(ProteinStructureFeature, "ProteinStructure")
