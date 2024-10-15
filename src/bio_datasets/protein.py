"""Defines protein objects that are lightweight wrappers around Biotite's AtomArray and AtomArrayStack.

This library is not intended to be a general-purpose library for protein structure analysis.
We simply wrap Biotite's AtomArray and AtomArrayStack to offer a few convenience methods
for dealing with protein structures in an ML context.
"""

from typing import List, Union

import biotite.structure as bs
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure.filter import filter_amino_acids
from biotite.structure.info.groups import amino_acid_names
from biotite.structure.residues import get_residue_starts

AA_LETTERS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
AA_NAMES = [ProteinSequence.convert_letter_1to3(aa) for aa in AA_LETTERS]
AA_LETTER_WITH_X = AA_LETTERS + ["X"]
AA_LETTER_WITH_X_AND_GAP = AA_LETTER_WITH_X + ["-"]
AA_LETTER_TO_INDEX = {aa: i for i, aa in enumerate(AA_LETTERS)}
AA_NAME_TO_INDEX = {aa: i for i, aa in enumerate(AA_NAMES)}
AA_LETTER_TO_INDEX_WITH_X = {aa: i for i, aa in enumerate(AA_LETTER_WITH_X)}
AA_LETTER_TO_INDEX_WITH_X_AND_GAP = {
    aa: i for i, aa in enumerate(AA_LETTER_WITH_X_AND_GAP)
}

BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def get_aa_index(residue_letters: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(AA_LETTERS, residue_letters)
    values = np.arange(len(AA_LETTERS))[indices]
    return values


def filter_atom_names(array, atom_names):
    return np.isin(array.atom_name, atom_names)


def filter_backbone(array):
    """
    Filter all peptide backbone atoms of one array.

    N, CA, C and O

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where an atom
        is a part of the peptide backbone.
    """

    return filter_atom_names(array, BACKBONE_ATOMS) & filter_amino_acids(array)


# TODO: add support for batched application of these functions (i.e. to multiple proteins at once)
class Protein:

    """A single protein chain."""

    def __init__(
        self,
        atoms: bs.AtomArray,
    ):
        """
        Parameters
        ----------
        atoms : AtomArray
            The atoms of the protein.
        """
        aa_mask = np.isin(atoms.res_name, amino_acid_names())
        assert np.all(aa_mask), "Protein must only contain amino acids"
        assert np.unique(atoms.chain_id).size == 1, "Protein must be a single chain"
        self.atoms = atoms
        self._aa_index = None
        self._residue_starts = get_residue_starts(self.atoms)
        self.chain_id = atoms.chain_id[0]

    @property
    def aa_index(self) -> np.ndarray:
        if self._aa_index is None:
            self._aa_index = get_aa_index(self.atoms.res_name[self._residue_starts])
        return self._aa_index

    def atom_positions(self, atom_names: Union[str, List[str]]) -> np.ndarray:
        if isinstance(atom_names, str):
            return self.atoms.coord[filter_atom_names(self.atoms, [atom_names])]
        else:
            atom_positions = []
            for atom_name in atom_names:
                atom_mask = filter_atom_names(self.atoms, [atom_name])
                atom_positions.append(self.atoms.coord[atom_mask])
            return np.stack(atom_positions, axis=1)  # n, num_atoms, 3

    def backbone_positions(self) -> np.ndarray:
        return self.atom_positions(BACKBONE_ATOMS)

    def backbone(self) -> "Protein":
        return Protein(self.atoms[filter_backbone(self.atoms)])


class ProteinComplex(Protein):
    """A protein complex."""

    def __init__(self, proteins: List[Protein]):
        self._chain_ids = [prot.chain_id for prot in proteins]
        self._proteins_lookup = {prot.chain_id: prot for prot in proteins}
        self.atoms = sum([prot.atoms for prot in proteins], bs.AtomArray())

    @property
    def get_chain(self, chain_id: str) -> Protein:
        return self._proteins_lookup[chain_id]
