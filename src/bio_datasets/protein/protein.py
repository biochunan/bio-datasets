"""Defines protein objects that are lightweight wrappers around Biotite's AtomArray and AtomArrayStack.

This library is not intended to be a general-purpose library for protein structure analysis.
We simply wrap Biotite's AtomArray and AtomArrayStack to offer a few convenience methods
for dealing with protein structures in an ML context.
"""

from typing import List, Optional, Union

import biotite.structure as bs
import numpy as np
from biotite.structure.filter import filter_amino_acids
from biotite.structure.residues import get_residue_starts

from bio_datasets.np_utils import map_categories_to_indices

from .constants import (
    RESTYPE_ATOM37_TO_ATOM14,
    atom_types,
    residue_atoms_ordered,
    resnames,
    restype_name_to_atom14_names,
    restypes,
)

BACKBONE_ATOMS = ["N", "CA", "C", "O"]
ALL_EXTRA_FIELDS = ["occupancy", "b_factor", "atom_id", "charge"]


def get_residue_starts_mask(
    atoms: bs.AtomArray, residue_starts: Optional[np.ndarray] = None
) -> np.ndarray:
    if residue_starts is None:
        residue_starts = get_residue_starts(atoms)
    mask = np.zeros(len(atoms), dtype=bool)
    mask[residue_starts] = True
    return mask


def tile_residue_annotation_to_atoms(
    atoms: bs.AtomArray, residue_annotation: np.ndarray, residue_starts: np.ndarray
) -> np.ndarray:
    # use residue index as cumsum of residue starts
    assert len(residue_annotation) == len(residue_starts)
    residue_index = np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1
    return residue_annotation[residue_index]


def get_relative_atom_indices_mapping() -> np.ndarray:
    """
    Get a mapping from atom37 index to expected index for a given residue.
    """
    all_atom_indices_mapping = []
    for resname in resnames:
        if resname == "UNK":
            residue_atom_list = ["N", "CA", "C", "O"]
        else:
            residue_atom_list = residue_atoms_ordered[resname]
        atom_indices_mapping = []
        for atom in atom_types:
            if atom in residue_atom_list:
                relative_index = residue_atom_list.index(atom)
                atom_indices_mapping.append(relative_index)
            else:
                atom_indices_mapping.append(-100)
        all_atom_indices_mapping.append(np.array(atom_indices_mapping))
    return np.stack(all_atom_indices_mapping, axis=0)


def decode_aa_index(aa_index: np.ndarray) -> np.ndarray:
    return "".join(np.array(restypes)[aa_index])


ATOM37_TO_RELATIVE_ATOM_INDEX_MAPPING = get_relative_atom_indices_mapping()  # (21, 37)
STANDARD_ATOMS_BY_RESIDUE = np.asarray(
    [restype_name_to_atom14_names[resname] for resname in resnames]
)  # (21, 14) indexable atom name strings
RESIDUE_SIZES = np.array(
    [len(residue_atoms_ordered[resname]) for resname in resnames[:-1]] + [4]
)  # bb only for UNK


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


def set_annotation_at_masked_atoms(
    atoms: bs.AtomArray, annot_name: str, new_annot: np.ndarray
):
    assert "mask" in atoms._annot
    atoms.add_annotation(annot_name, dtype=new_annot.dtype)
    if len(new_annot) != len(atoms):
        assert len(new_annot) == np.sum(atoms.mask)
        getattr(atoms, annot_name)[atoms.mask] = new_annot
    else:
        getattr(atoms, annot_name)[atoms.mask] = new_annot[atoms.mask]


def create_complete_atom_array_from_aa_index(
    aa_index: np.ndarray,
    chain_id: Union[str, np.ndarray],
    extra_fields: Optional[List[str]] = None,
    backbone_only: bool = False,
):
    """
    Populate annotations from aa_index, assuming all atoms are present.
    """
    if backbone_only:
        residue_sizes = [4] * len(aa_index)
    else:
        residue_sizes = RESIDUE_SIZES[
            aa_index
        ]  # (n_residues,) NOT (n_atoms,) -- add 1 to account for OXT
    if isinstance(chain_id, str) and not backbone_only:
        residue_sizes[-1] += 1  # final OXT
    else:
        if not backbone_only:
            assert len(chain_id) == len(residue_sizes)
            final_residue_in_chain = chain_id != np.concatenate(
                [chain_id[1:], ["ZZZZ"]]
            )
            residue_sizes[final_residue_in_chain] += 1
    residue_starts = np.concatenate(
        [[0], np.cumsum(residue_sizes)[:-1]]
    )  # (n_residues,)
    new_atom_array = bs.AtomArray(length=np.sum(residue_sizes))
    residue_index = (
        np.cumsum(get_residue_starts_mask(new_atom_array, residue_starts)) - 1
    )

    full_annot_names = []
    if isinstance(chain_id, str):
        new_atom_array.chain_id = np.full(len(new_atom_array), chain_id)
    else:
        new_atom_array.chain_id = chain_id[residue_index]
    full_annot_names.append("chain_id")

    # final atom in chain is OXT
    relative_atom_index = np.arange(len(new_atom_array)) - residue_starts[residue_index]
    atom_names = new_atom_array.atom_name
    if not backbone_only:
        oxt_mask = new_atom_array.chain_id != np.concatenate(
            [new_atom_array.chain_id[1:], ["ZZZZ"]]
        )
        atom_names[oxt_mask] = "OXT"
    else:
        oxt_mask = np.zeros(len(new_atom_array), dtype=bool)
    new_atom_array.set_annotation("aa_index", aa_index[residue_index])
    atom_names[~oxt_mask] = STANDARD_ATOMS_BY_RESIDUE[
        new_atom_array.aa_index[~oxt_mask],
        relative_atom_index[~oxt_mask],
    ]
    new_atom_array.set_annotation("atom_name", atom_names)
    new_atom_array.set_annotation(
        "res_name", np.array(resnames)[new_atom_array.aa_index]
    )
    new_atom_array.set_annotation("residue_index", residue_index)
    new_atom_array.set_annotation("res_id", residue_index + 1)
    full_annot_names += ["atom_name", "aa_index", "res_name", "residue_index", "res_id"]
    if extra_fields is not None:
        for f in extra_fields:
            new_atom_array.add_annotation(
                f, dtype=float if f in ["occupancy", "b_factor"] else int
            )
    return new_atom_array, residue_starts, full_annot_names


# TODO: add support for batched application of these functions (i.e. to multiple proteins at once)
class Protein:

    """A single protein chain.

    N.B. whereas the underlying biotite atom array exposes atom-level annotations,
    this class exposes residue level annotations.

    TODO: add option to disable standardisation
    TODO: add option to fill missing residues (especially useful for cif files)
    """

    def __init__(
        self,
        atoms: bs.AtomArray,
        verbose: bool = False,
        backbone_only: bool = False,
    ):
        """
        Parameters
        ----------
        atoms : AtomArray
            The atoms of the protein.
        """
        self.backbone_only = backbone_only
        atoms = atoms[filter_amino_acids(atoms)]
        assert np.unique(atoms.chain_id).size == 1, "Only a single chain is supported"
        self.atoms, self._residue_starts = self.standardise_atoms(
            atoms, verbose=verbose, backbone_only=backbone_only
        )
        self._standardised = True

    @staticmethod
    def set_atom_annotations(atoms, residue_starts):
        # convert selenium to sulphur
        mse_selenium_mask = (atoms.res_name == "MSE") & (atoms.atom_name == "SE")
        sec_selenium_mask = (atoms.res_name == "SEC") & (atoms.atom_name == "SE")
        atoms.atom_name[mse_selenium_mask] = "SD"
        atoms.atom_name[sec_selenium_mask] = "SG"
        atoms.res_name[atoms.res_name == "MSE"] = "MET"
        atoms.res_name[atoms.res_name == "SEC"] = "CYS"

        atoms.set_annotation(
            "atom37_index", map_categories_to_indices(atoms.atom_name, atom_types)
        )
        atoms.set_annotation("aa_index", get_aa_index(atoms.res_name))
        atoms.set_annotation(
            "residue_index",
            np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1,
        )
        return atoms

    @staticmethod
    def standardise_atoms(
        atoms,
        residue_starts: Optional[np.ndarray] = None,
        verbose: bool = False,
        backbone_only: bool = False,
    ):
        """We want all atoms to be present, with nan coords if any are missing.

        We also want to ensure that atoms are in the correct order.

        We can do this in a vectorised way by calculating the expected index of each atom,
        creating a new atom array with number of atoms equal to the expected number of atoms,
        and then filling in the present atoms in the new array according to the expected index.

        This standardisation ensures that methods like `backbone_positions`,`to_atom14`,
        and `to_atom37` can be applied safely downstream.
        """
        if residue_starts is None:
            residue_starts = get_residue_starts(atoms)

        atoms = Protein.set_atom_annotations(atoms, residue_starts)
        # first we get an array of atom indices for each residue (i.e. a mapping from atom37 index to expected index
        # then we index into this array to get the expected index for each atom
        expected_relative_atom_indices = ATOM37_TO_RELATIVE_ATOM_INDEX_MAPPING[
            atoms.aa_index, atoms.atom37_index
        ]
        final_residue_in_chain = atoms.chain_id[residue_starts] != np.concatenate(
            [atoms.chain_id[residue_starts][1:], ["ZZZZ"]]
        )
        final_residue_in_chain = tile_residue_annotation_to_atoms(
            atoms, final_residue_in_chain, residue_starts
        )
        oxt_mask = (atoms.atom_name == "OXT") & final_residue_in_chain
        if np.any(oxt_mask):
            expected_relative_atom_indices[oxt_mask] = (
                ATOM37_TO_RELATIVE_ATOM_INDEX_MAPPING[atoms.aa_index[oxt_mask]].max()
                + 1
            )
        unexpected_atom_mask = expected_relative_atom_indices == -100
        if np.any(unexpected_atom_mask):
            unexpected_atoms = atoms.atom_name[unexpected_atom_mask]
            unexpected_residues = atoms.res_name[unexpected_atom_mask]
            unexpected_str = "\n".join(
                [
                    f"{res_name} {res_id} {atom_name}"
                    for res_name, res_id, atom_name in zip(
                        unexpected_residues,
                        atoms.res_id[unexpected_atom_mask],
                        unexpected_atoms,
                    )
                ]
            )
            raise ValueError(
                f"At least one unexpected atom detected in a residue: {unexpected_str}.\n"
                f"HETATMs are not supported."
            )

        (
            new_atom_array,
            full_residue_starts,
            full_annot_names,
        ) = create_complete_atom_array_from_aa_index(
            atoms.aa_index[residue_starts],
            atoms.chain_id[residue_starts],
            extra_fields=[f for f in ALL_EXTRA_FIELDS if f in atoms._annot],
        )
        existing_atom_indices_in_full_array = (
            full_residue_starts[atoms.residue_index] + expected_relative_atom_indices
        ).astype(int)

        for annot_name, annot in atoms._annot.items():
            if annot_name in ["atom37_index", "mask"] or annot_name in full_annot_names:
                continue
            getattr(new_atom_array, annot_name)[
                existing_atom_indices_in_full_array
            ] = annot

        # set_annotation vs setattr: set_annotation adds to annot and verifies size
        new_atom_array.coord[existing_atom_indices_in_full_array] = atoms.coord
        # if we can create a res start index for each atom, we can assign the value based on that...
        assert len(full_residue_starts) == len(
            residue_starts
        ), f"Full residue starts: {full_residue_starts} and residue starts: {residue_starts} do not match"
        new_atom_array.set_annotation(
            "res_id", atoms.res_id[residue_starts][new_atom_array.residue_index]
        )  # override with auth res id
        new_atom_array.chain_id = atoms.chain_id[residue_starts][
            new_atom_array.residue_index
        ]

        new_atom_array.set_annotation(
            "atom37_index",
            map_categories_to_indices(new_atom_array.atom_name, atom_types),
        )
        assert np.all(
            new_atom_array.atom_name != ""
        ), "All atoms must be assigned a name"
        mask = np.zeros(len(new_atom_array), dtype=bool)
        mask[existing_atom_indices_in_full_array] = True
        missing_atoms_strings = [
            f"{res_name} {res_id} {atom_name}"
            for res_name, res_id, atom_name in zip(
                new_atom_array.res_name[~mask],
                new_atom_array.res_id[~mask],
                new_atom_array.atom_name[~mask],
            )
        ]
        if verbose:
            print("Filled in missing atoms:\n", "\n".join(missing_atoms_strings))
        new_atom_array.set_annotation("mask", mask)
        if backbone_only:
            # TODO: more efficient backbone only
            new_atom_array = new_atom_array[filter_backbone(new_atom_array)]
            full_residue_starts = get_residue_starts(new_atom_array)
        return new_atom_array, full_residue_starts

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
        assert (
            self._standardised
        ), "Atoms must be standardised before calculating backbone mask"
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

    def beta_carbon_coords(self) -> np.ndarray:
        has_beta_carbon = self.atoms.res_name != "GLY"
        beta_carbon_coords = np.zeros((self.num_residues, 3), dtype=np.float32)
        beta_carbon_coords[has_beta_carbon[self._residue_starts]] = self.atoms.coord[
            self._residue_starts[has_beta_carbon] + 4
        ]
        beta_carbon_coords[~has_beta_carbon[self._residue_starts]] = self.atoms.coord[
            self._residue_starts[~has_beta_carbon] + 1
        ]  # ca for gly
        return beta_carbon_coords

    def backbone_coords(self, atom_names: Optional[List[str]] = None) -> np.ndarray:
        assert all(
            [atom in BACKBONE_ATOMS + ["CB"] for atom in atom_names]
        ), "Invalid atom names"
        backbone_coords = self.atoms.coord[self.backbone_mask].reshape(
            -1, len(BACKBONE_ATOMS), 3
        )
        if atom_names is None:
            return backbone_coords
        else:
            backbone_atom_indices = [
                BACKBONE_ATOMS.index(atom) for atom in atom_names if atom != "CB"
            ]
            selected_coords = np.zeros(
                (len(backbone_coords), len(atom_names), 3), dtype=np.float32
            )
            selected_backbone_indices = [
                atom_names.index(atom) for atom in atom_names if atom != "CB"
            ]
            selected_coords[:, selected_backbone_indices] = backbone_coords[
                :, backbone_atom_indices
            ]
            if "CB" in atom_names:
                cb_index = atom_names.index("CB")
                selected_coords[:, cb_index] = self.beta_carbon_coords()
            return selected_coords

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
