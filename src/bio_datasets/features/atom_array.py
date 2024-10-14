"""Biotite atom array feature type for compatibility with HF datasets.

We write an Arrow ExtensionType for the AtomArray.
A couple of options for this:
1. Atom14
2. Backbone
3. Atom37
The issue with other formats is that the list of constituents could vary.
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pyarrow as pa
from biotite import structure as bs
from biotite.sequence import ProteinSequence
from biotite.structure.filter import filter_amino_acids
from biotite.structure.residues import get_residues
from datasets import Feature
from datasets.features.features import register_feature, string_to_arrow
from datasets.table import array_cast

from bio_datasets import constants as bio_constants

extra_annots = [
    "b_factor",
    "occupancy",
    "charge",
    "element",  # seems redundant
    "atom_id",
]


def get_sequence(struct: bs.AtomArray):
    residue_identities = get_residues(struct)[1]
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return seq


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

    return filter_atom_names(array, bio_constants.BACKBONE_ATOMS) & filter_amino_acids(
        array
    )


def atom_array_from_dict(d: dict) -> bs.AtomArray:
    sequence = d["sequence"]
    annots_keys = [k for k in d.keys() if k in extra_annots]
    if "backbone_coords" in d:
        backbone_coords = d["backbone_coords"]
        assert len(sequence) == len(d["backbone_coords"]["N"])
        atoms = []
        for res_ix, aa in enumerate(sequence):
            # TODO: better support for non-standard amino acids
            if aa == "U":
                aa = "C"
            if aa == "O":
                aa = "K"
            res_name = ProteinSequence.convert_letter_1to3(aa)
            for atom_name in bio_constants.BACKBONE_ATOMS:
                annots = {}
                for k in annots_keys:
                    annots[k] = d[k][res_ix]
                atom = bs.Atom(
                    coord=backbone_coords[atom_name][res_ix],
                    chain_id="A",
                    res_id=res_ix + 1,
                    res_name=res_name,
                    hetero=False,
                    atom_name=atom_name,
                    element=atom_name[0],
                    **annots,
                )
                atoms.append(atom)
        arr = bs.array(atoms)
        return arr
    elif "atom37_coords" in d:
        raise NotImplementedError("Atom37 not supported yet")
    else:
        raise ValueError("No coordinates found")


# n.b. metadata like chain_id, pdb_id, etc. should be stored separately
@dataclass
class _AtomArrayFeature(Feature):
    with_box: bool = False
    with_bonds: bool = False
    with_occupancy: bool = False
    with_b_factor: bool = False
    with_res_id: bool = False  # can be inferred...
    with_atom_id: bool = False
    with_charge: bool = False
    with_element: bool = False
    with_ins_code: bool = False
    with_hetero: bool = False

    @property
    def extra_fields(self):
        # values that can be passed to biotite load_structure
        extra_fields = []
        if self.with_occupancy:
            extra_fields.append("occupancy")
        if self.with_b_factor:
            extra_fields.append("b_factor")
        if self.with_atom_id:
            extra_fields.append("atom_id")
        if self.with_charge:
            extra_fields.append("charge")
        return extra_fields


@dataclass
class AtomArray(_AtomArrayFeature):
    """AtomArray [`Feature`] to read macromolecular atomic structure data from a PDB or CIF file.

    This feature stores the array directly as a pa struct (basically a dictionary of arrays),
    as defined in the AtomArrayExtensionType.

    Input: The BiomolecularStructureFile feature accepts as (encodeable) input (Q. where would 'input' typically occur):
    - A `biotite.structure.AtomArray` object.
    - TODO: a Biopython structure object
    - TODO: a file handler or file contents string?
    - A dictionary with the required keys:
        - sequence
        - backbone_coords: a dictionary with keys:
            - N
            - CA
            - C
            - O
        - atom37_coords: an array of shape (N, 37, 3)
        - atom37_mask: a boolean array of shape (N, 37)
        Only backbone_coords or atom37_coords + mask need to be provided, not both.
        All other keys are optional, but must correspond to fields in the AtomArrayExtensionType:
        - chain_id
        - res_id
        - ins_code
        - res_name
        - hetero
        - atom_name
        - box
        - bonds
        - occupancy
        - b_factor
        - atom_id
        - charge

    From biotite docs:
    Atom attributes:

    The following annotation categories are mandatory:

    =========  ===========  =================  =======================================
    Category   Type         Examples           Description
    =========  ===========  =================  =======================================
    chain_id   string (U4)  'A','S','AB', ...  Polypeptide chain
    res_id     int          1,2,3, ...         Sequence position of residue
    ins_code   string (U1)  '', 'A','B',..     PDB insertion code (iCode)
    res_name   string (U5)  'GLY','ALA', ...   Residue name
    hetero     bool         True, False        False for ``ATOM``, true for ``HETATM``
    atom_name  string (U6)  'CA','N', ...      Atom name
    element    string (U2)  'C','O','SE', ...  Chemical Element
    =========  ===========  =================  =======================================

    For all :class:`Atom`, :class:`AtomArray` and :class:`AtomArrayStack`
    objects these annotations are initially set with default values.
    Additionally to these annotations, an arbitrary amount of annotation
    categories can be added via :func:`add_annotation()` or
    :func:`set_annotation()`.
    The annotation arrays can be accessed either via the method
    :func:`get_annotation()` or directly (e.g. ``array.res_id``).

    The following annotation categories are optionally used by some
    functions:

    =========  ===========  =================   ============================
    Category   Type         Examples            Description
    =========  ===========  =================   ============================
    atom_id    int          1,2,3, ...          Atom serial number
    b_factor   float        0.9, 12.3, ...      Temperature factor
    occupancy  float        .1, .3, .9, ...     Occupancy
    charge     int          -2,-1,0,1,2, ...    Electric charge of the atom
    =========  ===========  =================   ============================

    Bond information can be associated to an :class:`AtomArray` or
    :class:`AtomArrayStack` by setting the ``bonds`` attribute with a
    :class:`BondList`.
    A :class:`BondList` specifies the indices of atoms that form chemical
    bonds.
    """

    requires_encoding: bool = True
    requires_decoding: bool = True
    decode: bool = True
    coords_dtype: str = "float32"
    bfactor_dtype: str = "float32"
    chain_id: Optional[
        str
    ] = None  # single chain id - means we will intepret structure as a single chain
    id: Optional[str] = None
    drop_sidechains: bool = False
    internal_coords_type: str = None  # foldcomp, idealised, or pnerf
    # Automatically constructed
    _type: str = field(
        default="AtomArray", init=False, repr=False
    )  # probably requires registered feature type5

    def _generate_array_dtype(self, dtype, shape):
        # source: datasets ArrayXDExtensionType
        dtype = string_to_arrow(dtype)
        for _ in reversed(shape):
            dtype = pa.list_(dtype)
            # Don't specify the size of the list, since fixed length list arrays have issues
            # being validated after slicing in pyarrow 0.17.1
        return dtype

    def __call__(self):
        fields = [
            pa.field(
                "coords", self._generate_array_dtype(self.coords_dtype, (None, 3))
            ),  # 2D array with shape (None, 3)
            pa.field("res_name", pa.list_(pa.utf8())),  # residue name
            pa.field("atom_name", pa.list_(pa.utf8())),  # CA, C, N, etc.
            pa.field("chain_id", pa.list_(pa.utf8()), nullable=True),
        ]
        if self.with_res_id:
            fields.append(pa.field("res_id", pa.list_(pa.int16())))
        if self.with_hetero:
            fields.append(pa.field("hetero", pa.list_(pa.bool_()), nullable=True))
        if self.with_ins_code:
            fields.append(pa.field("ins_code", pa.list_(pa.utf8()), nullable=True))
        if self.with_box:
            fields.append(
                pa.field("box", pa.list_(pa.list_(pa.float32(), 3), 3), nullable=True)
            )
        if self.with_bonds:
            fields.append(
                pa.field(
                    "bonds",
                    pa.list_(
                        pa.struct(
                            fields=[
                                pa.field("atom1_idx", pa.int32()),
                                pa.field("atom2_idx", pa.int32()),
                                pa.field("bond_type", pa.int8()),
                            ]
                        )
                    ),
                    nullable=True,
                ),
            )
        if self.with_occupancy:
            fields.append(pa.field("occupancy", pa.list_(pa.float16()), nullable=True))
        if self.with_b_factor:
            fields.append(
                pa.field(
                    "b_factor",
                    pa.list_(string_to_arrow(self.bfactor_dtype)),
                    nullable=True,
                )
            )
        if self.with_charge:
            fields.append(pa.field("charge", pa.list_(pa.int8()), nullable=True))
        if self.with_element:
            fields.append(pa.field("element", pa.list_(pa.utf8()), nullable=True))
        return pa.struct(fields)

    def cast_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Cast an Arrow array to the AtomArray arrow storage type.
        Fields need to be in the same order as in AtomArrayExtensionType.
        https://github.com/huggingface/datasets/blob/16a121d7821a7691815a966270f577e2c503473f/src/datasets/table.py#L1995

        The Arrow types that can be converted to the AtomArray pyarrow storage type are:

        - `pa.struct({...})`  - order doesn't matter

        Args:
            storage (`Union[pa.StringArray, pa.StructArray, pa.ListArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the AtomArray arrow storage type.
        """
        if not pa.types.is_struct(storage.type):
            raise ValueError(f"Expected struct type, got {storage.type}")

        # Initialize arrays for all fields in AtomArrayExtensionType
        fields = {}
        for i in range(storage.type.num_fields):
            field_name = storage.type.field(i).name
            fields[field_name] = storage.field(field_name)

        # Create a new StructArray with all the required fields
        storage = pa.StructArray.from_arrays(
            list(fields.values()), list(fields.keys()), mask=storage.is_null()
        )

        return array_cast(storage, self())

    @property
    def required_keys(self):
        required_keys = ["coords", "atom_name", "res_name", "chain_id"]
        if self.with_box:
            required_keys.append("box")
        if self.with_bonds:
            required_keys.append("bonds")
        return required_keys

    def encode_example(self, value: Union[bs.AtomArray, dict]) -> dict:
        if isinstance(value, dict):
            # if it's already encoded, we don't need to encode it again
            if all([attr in value for attr in self.required_keys]):
                return value
            value = atom_array_from_dict(value)
            return self.encode_example(value)
        elif isinstance(value, bs.AtomArray):
            if self.drop_sidechains:
                value = value[filter_backbone(value)]
            atom_array_struct = {
                "coords": value.coord,
                "res_name": np.array(
                    [
                        ProteinSequence.convert_letter_3to1(res_name)
                        for res_name in value.res_name
                    ]
                ),
                "atom_name": value.atom_name,
            }
            if self.chain_id is None:
                atom_array_struct["chain_id"] = value.chain_id
            else:
                atom_array_struct["chain_id"] = None
            for attr in [
                "box",
                "bonds",
                "occupancy",
                "b_factor",
                "atom_id",
                "charge",
                "element",
                "res_id",
                "ins_code",
                "hetero",
            ]:
                if getattr(self, f"with_{attr}"):
                    atom_array_struct[attr] = getattr(value, attr)

            return atom_array_struct
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

    def decode_example(self, value: dict, token_per_repo_id=None) -> bs.AtomArray:
        """
        def add_annotation(self, category, dtype):
        Add an annotation category, if not already existing.

        Initially the new annotation is filled with the *zero*
        representation of the given type.

        Parameters
        ----------
        category : str
            The annotation category to be added.
        dtype : type or str
            A type instance or a valid *NumPy* *dtype* string.
            Defines the type of the annotation

        See Also
        --------
        set_annotation

        Notes
        -----
        If the annotation category already exists, a compatible dtype is chosen,
        that is also able to represent the old values.
        if category not in self._annot:
            self._annot[str(category)] = np.zeros(self._array_length, dtype=dtype)
        elif np.can_cast(self._annot[str(category)].dtype, dtype):
            self._annot[str(category)] = self._annot[str(category)].astype(dtype)
        elif np.can_cast(dtype, self._annot[str(category)].dtype):
            # The existing dtype is more general
            pass
        else:
            raise ValueError(
                f"Cannot cast '{str(category)}' "
                f"with dtype '{self._annot[str(category)].dtype}' into '{dtype}'"
        """
        # TODO: optimise this...if we set format to numpy, everything is a numpy array which should be ideal
        length = len(value["res_name"])
        if "res_id" not in value:
            value["res_id"] = np.arange(length)
        arr = bs.AtomArray(length=length)
        value["res_name"] = [
            ProteinSequence.convert_letter_1to3(aa) for aa in value["res_name"]
        ]
        coords = np.stack(value.pop("coords"))
        arr.coord = coords
        for key, value in value.items():
            arr.set_annotation(key, value)
        if self.chain_id is not None:
            arr.set_annotation("chain_id", np.array([self.chain_id] * length))
        return arr
