"""Defines protein objects that are lightweight wrappers around Biotite's AtomArray and AtomArrayStack.

This library is not intended to be a general-purpose library for protein structure analysis.
We simply wrap Biotite's AtomArray and AtomArrayStack to offer a few convenience methods
for dealing with protein structures in an ML context.
"""

from typing import List, Union

import biotite.structure as bs
import numpy as np
from biotite.structure.filter import filter_amino_acids
from biotite.structure.info.groups import amino_acid_names
from biotite.structure.residues import get_residue_starts

from .constants import atom_order, resnames

BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def get_aa_index(res_name: np.ndarray) -> np.ndarray:
    # n.b. resnames are sorted in alphabetical order, apart from UNK
    indices = np.searchsorted(np.array(resnames[:-1]), res_name)
    values = np.arange(len(resnames[:-1]))[indices]
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
        self._validate_atoms(atoms)
        self.atoms = atoms
        self._atom_type_index = getattr(
            atoms, "atom_typeindex", None
        )  # convenient to work with a fixed set of atoms - can assume atom37 for now
        self._aa_index = getattr(atoms, "aa_index", None)
        self._residue_index = getattr(
            atoms, "residue_index", None
        )  # internal numbering
        self._residue_starts = get_residue_starts(self.atoms)
        self.chain_id = atoms.chain_id[0]

    @property
    def aa_index(self) -> np.ndarray:
        if self._aa_index is None:
            self._aa_index = get_aa_index(self.atoms.res_name[self._residue_starts])
            # ensures we pass on when cloning
            self.atoms.set_annotation("aa_index", self._aa_index)
        return self._aa_index

    @property
    def residue_index(self) -> np.ndarray:
        if self._residue_index is None:
            self._residue_index = np.arange(self.num_residues)
            # ensures we pass on when cloning
            self.atoms.set_annotation("residue_index", self._residue_index)
        return self._residue_index

    @property
    def atom_index(self) -> np.ndarray:
        if self._atom_type_index is None:
            unique_atom_names, indices = np.unique(
                self.atoms.atom_name, return_inverse=True
            )
            atom37_indices = [atom_order[name] for name in unique_atom_names]
            self._atom_type_index = np.array(atom37_indices)[indices]
            # ensures we pass on when cloning
            self.atoms.set_annotation("atom_type_index", self._atom_type_index)
        return self._atom_type_index

    @property
    def num_residues(self) -> int:
        return len(self._residue_starts)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.atoms[key]
        else:
            return Protein(self.atoms[key])

    def _validate_standardise_atoms(self, atoms: bs.AtomArray):
        """We want all backbone atoms to be present, with nan coords if any are missing.

        This validation ensures that methods like `backbone_positions`,`to_atom14`, and `to_atom37` can be applied safely downstream.
        """
        aa_mask = np.isin(atoms.res_name, amino_acid_names())
        assert np.all(aa_mask), "Protein must only contain amino acids"
        assert np.unique(atoms.chain_id).size == 1, "Protein must be a single chain"
        atom_counts = np.sum(
            atoms.atom_name[:, None] == np.array(BACKBONE_ATOMS), axis=0
        )
        for atom_name, atom_count in zip(BACKBONE_ATOMS, atom_counts):
            if atom_count != self.num_residues:
                print(f"Warning: missing {atom_name} atoms detected, filling wiht NaNs")
                self._fill_missing_atoms(atoms, atom_name)

    def _fill_missing_atoms(self, atoms, atom_name):
        residues_with_atom = np.unique(atoms.res_id[atoms.atom_name == atom_name])
        missing_residues = np.setdiff1d(self.residue_index, residues_with_atom)
        # to vectorise: we need to create a missing atoms mask for each atom
        # similarly to reorder, we should be able to create an expected atom index for each
        # current atom. in fact if we can do this, we can do fill and reorder in a single pass
        raise NotImplementedError("Not implemented yet")
        # for res_id in missing_residues:
        #     residue_start = self._residue_starts[res_id]
        #     res_name = atoms.res_name[residue_start]
        #     atom_index = atom_index_in_residue(res_name, atom_name)
        #     new_atom = Atom()
        #     self.atoms = self.atoms[:residue_start+atom_index] + new_atom + self.atoms[residue_start+atom_index:]

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

    def to_atom14(self) -> np.ndarray:
        raise NotImplementedError("Not implemented yet")

    def to_atom37(self) -> np.ndarray:
        raise NotImplementedError("Not implemented yet")

    def distances(self, atom_names: Union[str, List[str]], nan_fill=None) -> np.ndarray:
        # TODO: handle nans
        raise NotImplementedError("Not implemented yet")

    def contacts(self, atom_name: str = "CA", threshold: float = 8.0) -> np.ndarray:
        return self.distances(atom_name, nan_fill="max") < threshold
