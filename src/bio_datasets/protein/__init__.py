from .complex import ProteinComplex
from .protein import Protein


def load_example_protein():
    import os

    from biotite.structure.io.pdb import PDBFile

    pdb_atom_array = PDBFile.read(os.path.join("tests", "1qys.pdb")).get_structure(
        model=1
    )
    return Protein(pdb_atom_array)
