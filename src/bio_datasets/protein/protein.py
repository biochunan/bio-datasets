"""Defines protein objects that are lightweight wrappers around Biotite's AtomArray and AtomArrayStack.

This library is not intended to be a general-purpose library for protein structure analysis.
We simply wrap Biotite's AtomArray and AtomArrayStack to offer a few convenience methods
for dealing with protein structures in an ML context.
"""

from typing import List, Optional, Union

import biotite.structure as bs
import numpy as np
from biotite.structure.residues import get_residue_starts

from bio_datasets.np_utils import map_categories_to_indices

from .constants import RESTYPE_ATOM37_TO_ATOM14, atom_types, residue_atoms, resnames

BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def get_residue_starts_mask(
    atoms: bs.AtomArray, residue_starts: Optional[np.ndarray] = None
) -> np.ndarray:
    if residue_starts is None:
        residue_starts = get_residue_starts(atoms)
    mask = np.zeros(len(atoms), dtype=bool)
    mask[residue_starts] = True
    return mask


def get_relative_atom_indices_mapping(atom_names: List[str]) -> np.ndarray:
    """
    Get a mapping from atom37 index to expected index for a given residue.
    """
    all_atom_indices_mapping = []
    for resname in resnames:
        if resname == "UNK":
            residue_atom_list = ["N", "CA", "C", "O"]
        else:
            residue_atom_list = residue_atoms[resname]
        atom_indices_mapping = []
        for atom in atom_types:
            if atom in residue_atom_list:
                atom_indices_mapping[resname, atom] = residue_atom_list.index(atom)
            else:
                atom_indices_mapping[resname, atom] = -100
        all_atom_indices_mapping.append(np.array(atom_indices_mapping))
    return np.stack(all_atom_indices_mapping, axis=0)


RELATIVE_ATOM_INDICES_MAPPING = get_relative_atom_indices_mapping(atom_types)
RESIDUE_SIZES = np.array([len(residue_atoms[resname]) for resname in resnames])


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

    """A single protein chain.

    N.B. whereas the atom array exposes atom level annotations,
    this class exposes residue level annotations.
    """

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
        self._set_atom_annotations(atoms)
        self.atoms, self._residue_starts = self._standardise_atoms(atoms)

    def _set_atom_annotations(self, atoms):
        residue_starts = get_residue_starts(atoms)
        atoms.set_annotation(
            "atom37_index", map_categories_to_indices(atoms.atom_name, atom_types)
        )
        atoms.set_annotation("aa_index", get_aa_index(atoms.res_name))
        atoms.set_annotation(
            "residue_index", np.cumsum(get_residue_starts_mask(atoms, residue_starts))
        )

    @property
    def chain_id(self):
        return self.atoms.chain_id[0]

    @property
    def residue_index(self):
        return self.atoms["residue_index"][self._residue_starts]

    @property
    def aa_index(self):
        return self.atoms["aa_index"][self._residue_starts]

    @property
    def num_residues(self) -> int:
        return len(self._residue_starts)

    def __len__(self):
        return self.num_residues  # n.b. -- not equal to len(self.atoms)

    @property
    def backbone_mask(self):
        # assumes standardised atoms
        residue_start_mask = np.zeros(len(self.atoms), dtype=bool)
        residue_start_mask[self._residue_starts] = True
        offsets = len(BACKBONE_ATOMS) * self.atoms.residue_index
        return (~residue_start_mask).cumsum() - offsets <= 3

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.atoms[key]
        else:
            return Protein(self.atoms[key])

    def _standardise_atoms(self):
        """We want all atoms to be present, with nan coords if any are missing.

        We also want to ensure that atoms are in the correct order.

        We can do this in a vectorised way by calculating the expected index of each atom,
        created a new atom array with number of atoms equal to the expected number of atoms,
        and then filling in the present atoms in the new array according to the expected index.

        This validation ensures that methods like `backbone_positions`,`to_atom14`, and `to_atom37` can be applied safely downstream.
        """
        import time

        t0 = time.time()
        # first we get an array of atom indices for each residue (i.e. a mapping from atom37 index to expected index
        # then we index into this array to get the expected index for each atom
        expected_relative_atom_indices = RELATIVE_ATOM_INDICES_MAPPING[
            self.atoms.residue_index, self.atoms.atom37_index
        ]
        if np.any(expected_relative_atom_indices == -100):
            raise ValueError(
                "At least one unexpected atom detected in a residue. HETATMs are not supported."
            )

        # we need to add these indices to expected residue starts
        residue_sizes = RESIDUE_SIZES[
            self.residue_index
        ]  # (n_residues,) NOT (n_atoms,)
        residue_starts = np.concatenate(
            [0, np.cumsum(residue_sizes)[:-1]]
        )  # (n_residues,)
        expected_atom_indices = (
            residue_starts[self.atoms.residue_index] + expected_relative_atom_indices
        )

        new_atom_array = bs.AtomArray(length=np.sum(residue_sizes))
        for annot_name, annot in self.atoms.annotations.items():
            new_annot = np.full(len(new_atom_array), np.nan)
            new_annot[expected_atom_indices] = annot
            new_atom_array.set_annotation(annot_name, new_annot)
        return new_atom_array, residue_starts

    def atom_coords(self, atom_names: Union[str, List[str]]) -> np.ndarray:
        if isinstance(atom_names, str):
            return self.atoms.coord[filter_atom_names(self.atoms, [atom_names])]
        else:
            atom_positions = []
            for atom_name in atom_names:
                atom_mask = filter_atom_names(self.atoms, [atom_name])
                atom_positions.append(self.atoms.coord[atom_mask])
            return np.stack(atom_positions, axis=1)  # n, num_atoms, 3

    def backbone_coords(self) -> np.ndarray:
        return self.atoms.coord[self.backbone_mask].reshape(-1, len(BACKBONE_ATOMS), 3)

    def backbone(self) -> "Protein":
        return Protein(self.atoms[self.backbone_mask])

    def atom14_coords(self) -> np.ndarray:
        atom14_coords = np.full((len(self.num_residues), 14, 3), np.nan)
        atom14_index = RESTYPE_ATOM37_TO_ATOM14[
            self.atoms.residue_index, self.atoms.atom37_index
        ]
        atom14_coords[self.atoms.residue_index, atom14_index] = self.atoms.coord
        return atom14_coords

    def atom37_coords(self) -> np.ndarray:
        # since we have standardised the atoms we can just return standardised atom37 indices for each residue
        atom37_coords = np.full((len(self.num_residues), len(atom_types), 3), np.nan)
        atom37_coords[
            self.atoms.residue_index, self.atoms.atom37_index
        ] = self.atoms.coord
        return atom37_coords

    def distances(
        self,
        atom_names: Union[str, List[str]],
        nan_fill=None,
        multi_atom_calc_type: str = "max",
    ) -> np.ndarray:
        # TODO: handle nans
        backbone_coords = self.backbone_coords()
        if atom_names in BACKBONE_ATOMS:
            at_index = BACKBONE_ATOMS.index(atom_names)
            dists = np.sqrt(
                np.sum(
                    (
                        backbone_coords[None, :, at_index, :]
                        - backbone_coords[:, None, at_index, :]
                    )
                    ** 2,
                    axis=-1,
                )
            )
        else:
            raise NotImplementedError(
                "Muliple atom distance calculations not yet supported"
            )
        if nan_fill is not None:
            if isinstance(nan_fill, float) or isinstance(nan_fill, int):
                dists = np.nan_to_num(dists, nan=nan_fill)
            elif nan_fill == "max":
                max_dist = np.nanmax(dists, axis=-1)
                dists = np.nan_to_num(dists, nan=max_dist)
            else:
                raise ValueError(
                    f"Invalid nan_fill: {nan_fill}. Please specify a float or int."
                )
        return dists

    def contacts(self, atom_name: str = "CA", threshold: float = 8.0) -> np.ndarray:
        return self.distances(atom_name, nan_fill="max") < threshold
