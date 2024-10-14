"""Biotite atom array feature type for compatibility with HF datasets.

We write an Arrow ExtensionType for the AtomArray.
A couple of options for this:
1. Atom14
2. Backbone
3. Atom37
The issue with other formats is that the list of constituents could vary.
"""
import os
import uuid
from dataclasses import dataclass, field
from io import BufferedIOBase, BytesIO, StringIO
from os import PathLike
from typing import Any, ClassVar, Optional, Union

import numpy as np
import pyarrow as pa
from biotite import structure as bs
from biotite.sequence import ProteinSequence
from biotite.structure import get_chains
from biotite.structure.filter import filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.residues import get_residues
from datasets import Feature, config
from datasets.download import DownloadConfig
from datasets.features.features import string_to_arrow
from datasets.table import array_cast
from datasets.utils.file_utils import is_local_path, xopen, xsplitext
from datasets.utils.py_utils import string_to_dict

from bio_datasets import config as bio_config
from bio_datasets import constants as bio_constants

if bio_config.FOLDCOMP_AVAILABLE:
    import foldcomp


extra_annots = [
    "b_factor",
    "occupancy",
    "charge",
    "element",  # seems redundant
    "atom_id",
]


def filter_chains(structure, chain_ids):
    # TODO: double-check numeric chain id is ok...
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    if chain_ids is None:
        return structure
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


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


def load_structure(
    fpath_or_handler,
    format="pdb",
    model: int = 1,
    extra_fields=None,
):
    """
    TODO: support foldcomp format, binary cif format
    TODO: support model choice / multiple models (multiple conformations)
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if format == "cif":
        pdbxf = CIFFile.read(fpath_or_handler)
        structure = pdbxf.get_structure(
            model=model,
            extra_fields=extra_fields,
        )
    elif format == "pdb":
        pdbf = PDBFile.read(fpath_or_handler)
        structure = pdbf.get_structure(
            model=model,
            extra_fields=extra_fields,
        )
    elif format == "fcz":
        if not bio_config.FOLDCOMP_AVAILABLE:
            raise ImportError("Foldcomp is not installed. Please install it with `pip install foldcomp`")
        import foldcomp
        if is_open_compatible(fpath_or_handler):
            with open(fpath_or_handler, "rb") as fcz:
                fcz_binary = fcz.read()
        elif isinstance(fpath_or_handler, BufferedIOBase):
            fcz_binary = fcz.read()
        else:
            raise ValueError(f"Unsupported file type: expected path or bytes handler")
        (_, pdb_str) = foldcomp.decompress(fcz_binary)
        lines = pdb_str.splitlines()
        pdbf = PDBFile()
        pdbf.lines = lines
        structure = pdbf.get_structure(
            pdbf,
            model=model,
            extra_fields=extra_fields,
        )
    else:
        raise ValueError(f"Unsupported file format: {format}")

    return structure


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


def encode_biotite_atom_array(array: bs.AtomArray, encode_with_foldcomp: bool = False, name: Optional[str] = None) -> str:
    """
    Encode a biotite AtomArray to pdb string bytes.

    TODO: support foldcomp encoding
    """
    pdb = pdb.PDBFile()
    pdb.set_structure(array)
    contents = "\n".join(pdb.lines) + "\n"
    if encode_with_foldcomp:
        import foldcomp
        if name is None:
            name = getattr(array, "name", str(uuid.uuid4()))
        return foldcomp.compress(name, contents)
    else:
        return contents.encode()


def is_open_compatible(file):
    return isinstance(file, (str, bytes, PathLike))


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
        elif isinstance(value, (str, os.PathLike)):
            if os.path.exists(value):
                pass
        elif isinstance(value, bytes):
            # assume it encodes file contents.
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


@dataclass
class Structure(_AtomArrayFeature):
    """Structure [`Feature`] to read (bio)molecular atomic structure data from supported file types.
    The file contents are serialized as bytes, file path and file type within an Arrow table.
    The file contents are automatically decoded to a biotite AtomArray (if mode=="array") or a
    Biopython structure (if mode=="structure") when loading data from the dataset.

    This is similar to the Image feature in the HF datasets library.

    - AtomArray documentation: https://www.biotite-python.org/latest/apidoc/biotite.structure.AtomArray.html#biotite.structure.AtomArray
    - Structure documentation: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ#the-structure-object

    Input: The BiomolecularStructureFile feature accepts as (encodeable) input (Q. where would 'input' typically occur):
    - A `str`: Absolute path to the structure file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the structure file to the archive file.
        - `bytes`: Bytes of the structure file.

      This is useful for archived files with sequential access.
    - A `biotite.structure.AtomArray` object.
    - TODO: a Biopython structure object
    - TODO: a file handler or file contents string?

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the structure data. If `False`,
            returns the underlying dictionary in the format `{"path": structure_path, "bytes": structure_bytes, "type": structure_type}`.
    """

    requires_encoding: bool = True
    requires_decoding: bool = True
    decode: bool = True
    id: Optional[str] = None
    encode_with_foldcomp: bool = False
    mode: str = "array"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string(), "type": pa.string()})
    _type: str = field(default="Structure", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bs.AtomArray]) -> dict:
        """Encode example into a format for Arrow.

        This determines what gets written to the Arrow file.
        """
        if isinstance(value, str):
            if file_type is None:
                file_type = xsplitext(value)[1][1:].lower()
            return {"path": value, "bytes": None, "type": file_type}
        elif isinstance(value, bytes):
            # just assume pdb format for now
            return {"path": None, "bytes": value, "type": "pdb"}
        elif isinstance(value, bs.AtomArray):
            return {
                "path": None,
                "bytes": encode_biotite_atom_array(
                    value,
                    encode_with_foldcomp=self.encode_with_foldcomp,
                ),
                "type": "pdb" if not self.encode_with_foldcomp else "fcz",
            }
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            file_type = value.get("type", None)
            path = value["path"]
            if file_type is None:
                file_type = xsplitext(path)[1][1:].lower()
            # we set "bytes": None to not duplicate the data if they're already available locally
            # (this assumes invocation in what context?)
            return {"bytes": None, "path": path, "type": file_type or "pdb"}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the Structure bytes, and path is optionally used to infer the Structure format using the file extension
            file_type = value.get("type")
            path = value.get("path")
            if file_type is None and path is not None:
                file_type = xsplitext(path)[1][1:].lower()
            return {"bytes": value.get("bytes"), "path": path, "type": file_type}
        else:
            raise ValueError(
                f"A structure sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> "bs.AtomArray":
        """Decode example structure file into AtomArray data.

        Args:
            value (`str` or `dict`):
                A string with the absolute structure file path, a dictionary with
                keys:

                - `path`: String with absolute or relative structure file path.
                - `bytes`: The bytes of the structure file.
                - `type`: The type of the structure file (e.g. "pdb", "cif", "fcz")
                Must be not None.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                structure files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `biotite.AtomArray`
        """
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use Structure(decode=True) instead."
            )

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_, file_type = value["path"], value["bytes"], value["type"]
        if bytes_ is None:
            if path is None:
                raise ValueError(
                    f"A structure should have one of 'path' or 'bytes' but both are None in {value}."
                )
            else:
                if is_local_path(path):
                    atom_array = load_structure(
                        path,
                        format=file_type,
                        extra_fields=self.extra_fields,
                    )
                else:
                    source_url = path.split("::")[-1]
                    pattern = (
                        config.HUB_DATASETS_URL
                        if source_url.startswith(config.HF_ENDPOINT)
                        else config.HUB_DATASETS_HFFS_URL
                    )
                    try:
                        repo_id = string_to_dict(source_url, pattern)["repo_id"]
                        token = token_per_repo_id.get(repo_id)
                    except ValueError:
                        token = None
                    download_config = DownloadConfig(token=token)
                    with xopen(path, "r", download_config=download_config) as f:
                        atom_array = load_structure(
                            f,
                            format=file_type or "pdb",
                            extra_fields=self.extra_fields,
                        )

        else:
            if path is not None:
                if file_type == "fcz":
                    fhandler = BytesIO(bytes_)
                elif file_type == "pdb":
                    fhandler = StringIO(bytes_.decode())
                elif file_type == "cif":
                    fhandler = StringIO(bytes_.decode())
                else:
                    raise ValueError(
                        f"Unsupported file format: {file_type} for bytes input"
                    )
                atom_array = load_structure(
                    fhandler,
                    format=file_type,
                    extra_fields=self.extra_fields,
                )
            else:
                if file_type == "fcz":
                    _, pdb = foldcomp.decompress(bytes_)
                else:
                    pdb = bytes_.decode()
                contents = StringIO(pdb)
                # assume pdb format in bytes for now - is bytes only possible internally?
                atom_array = load_structure(
                    contents,
                    format="pdb",
                    extra_fields=self.extra_fields,
                )
        return atom_array

    def cast_storage(self, storage: pa.StructArray) -> pa.StructArray:
        if pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None]* len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array, storage.field("type")],
                names=["bytes", "path", "type"],
                mask=storage.is_null()
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage.type}")
        return array_cast(storage, self.pa_type)
