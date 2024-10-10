"""Biotite atom array feature type for compatibility with HF datasets.

We write an Arrow ExtensionType for the AtomArray.
A couple of options for this:
1. Atom14
2. Backbone
3. Atom37
The issue with other formats is that the list of constituents could vary.
"""
import os
from dataclasses import dataclass, field
from io import BufferedIOBase, BytesIO, StringIO
from os import PathLike
from typing import Any, ClassVar, Dict, Optional, Union

import biotite
import foldcomp
import numpy as np
import pyarrow as pa
from biotite import structure as bs
from biotite.sequence import ProteinSequence
from biotite.structure import filter_amino_acids, get_chains
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile
from datasets import config
from datasets.download.download_config import DownloadConfig
from datasets.table import array_cast
from datasets.utils.file_utils import is_local_path, xopen, xsplitext
from datasets.utils.py_utils import no_op_if_value_is_null, string_to_dict


def encode_biotite_atom_array(array: bs.AtomArray) -> str:
    """
    Encode a biotite AtomArray to pdb string bytes.

    TODO: support foldcomp encoding
    """
    pdb = pdb.PDBFile()
    pdb.set_structure(array)
    contents = "\n".join(pdb.lines) + "\n"
    return contents.encode()


def filter_atom_names(array, atom_names):
    return np.isin(array.atom_name, atom_names)


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


def is_open_compatible(file):
    return isinstance(file, (str, bytes, PathLike))


def load_structure(
    fpath_or_handler,
    format="pdb",
    model: int = 1,
    extra_fields=None,
    assembly_id=None,
    chain_ids=None,
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
            pdbxf,
            model=1,
            assembly_id=assembly_id,
            extra_fields=extra_fields,
            chain_ids=chain_ids,
        )
    elif format == "pdb":
        pdbf = PDBFile.read(fpath_or_handler)
        structure = pdbf.get_structure(
            pdbf,
            model=1,
            assembly_id=assembly_id,
            extra_fields=extra_fields,
            chain_ids=chain_ids,
        )
    elif format == "fcz":
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
            model=1,
            assembly_id=assembly_id,
            extra_fields=extra_fields,
            chain_ids=chain_ids,
        )
    else:
        raise ValueError(f"Unsupported file format: {format}")

    return structure


@dataclass
class AtomArray:
    """AtomArray [`Feature`] to read macromolecular atomic structure data from a PDB or CIF file.

    Input: The AtomArray feature accepts as input:
    - A `str`: Absolute path to the structure file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the structure file to the archive file.
        - `bytes`: Bytes of the structure file.

      This is useful for archived files with sequential access.
    - A `biotite.AtomArray` object.

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the structure data. If `False`,
            returns the underlying dictionary in the format `{"path": structure_path, "bytes": structure_bytes}`.

    # TODO: Examples:
    """

    decode: bool = True
    id: Optional[str] = None
    chain_ids: Optional[Union[str, List[str]]] = None
    assembly_id: Optional[int] = None  # TODO: properly understand this
    # Automatically constructed
    with_occupancy: bool = False
    with_b_factor: bool = False
    with_atom_id: bool = False
    with_charge: bool = False
    encode_with_foldcomp: bool = False
    foldcomp_anchor_residue_threshold: int = (
        25  # < 0.1 A RMSD (less than noise typically applied in generative models)
    )
    dtype: ClassVar[str] = "bs.AtomArrray"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="AtomArray", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    @property
    def extra_fields(self):
        extra_fields = []
        if self.with_occupancy:
            extra_fields.append("occupancy")
        if self.with_b_factor:
            extra_fields.append("b_factor")
        if self.with_atom_id:
            extra_fields.append("atom_id")
        if self.with_charge:
            extra_fields.append("charge")

    def encode_example(self, value: Union[str, bytes, biotite.AtomArray]) -> dict:
        """Encode example into a format for Arrow.

        For now we encode the pdb str.
        TODO: support encoding a dictionary of coords

        Args:
            value (`str`, `biotite.AtomArray`):
                Data passed as input to AtomArray feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, bytes):
            return {"path": None, "bytes": value}
        elif isinstance(value, bs.AtomArray):
            return {
                "path": None,
                "bytes": encode_biotite_atom_array(
                    value,
                    encode_with_foldcomp=self.encode_with_foldcomp,
                    foldcomp_anchor_residue_threshold=self.foldcomp_anchor_residue_threshold,
                ),
            }
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the Structure bytes, and path is used to infer the Structure format using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}
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

        path, bytes_ = value["path"], value["bytes"]
        file_format = xsplitext(path)[1][1:].lower() if path is not None else None
        if not file_format in ["pdb", "cif"]:
            raise ValueError(f"Unsupported file format: {file_format}")
        if bytes_ is None:
            if path is None:
                raise ValueError(
                    f"A structure should have one of 'path' or 'bytes' but both are None in {value}."
                )
            else:
                if is_local_path(path):
                    atom_array = load_structure(
                        path,
                        assembly_id=self.assembly_id,
                        chain_id=self.chain_id,
                        format=file_format,
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
                            assembly_id=self.assembly_id,
                            chain_id=self.chain_id,
                            format=file_format,
                            extra_fields=self.extra_fields,
                        )

        else:
            if path is not None:
                if file_format == "fcz":
                    fhandler = BytesIO(bytes_)
                elif file_format == "pdb":
                    fhandler = StringIO(bytes_.decode())
                elif file_format == "cif":
                    fhandler = StringIO(bytes_.decode())
                else:
                    raise ValueError(
                        f"Unsupported file format: {file_format} for bytes input"
                    )
                atom_array = load_structure(
                    fhandler,
                    assembly_id=self.assembly_id,
                    chain_id=self.chain_id,
                    format=file_format,
                    extra_fields=self.extra_fields,
                )
            else:
                contents = StringIO(bytes_.decode)
                # assume pdb format in bytes for now - is bytes only possible internally?
                atom_array = load_structure(
                    contents,
                    assembly_id=self.assembly_id,
                    chain_id=self.chain_id,
                    format="pdb",
                    extra_fields=self.extra_fields,
                )
        return atom_array

    def flatten(self) -> Union["FeatureType", Dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from datasets.features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(
        self, storage: Union[pa.StringArray, pa.StructArray, pa.BinaryArray]
    ) -> pa.StructArray:
        """
        Cast an Arrow array to the structure arrow storage type.
        The Arrow types that can be converted to the structure pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the structure bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter

        Args:
            storage (`Union[pa.StringArray, pa.StructArray, pa.ListArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Structure arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays(
                [bytes_array, storage], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [storage, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Embed structure files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.

        Returns:
            `pa.StructArray`: Array in the Structure arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """

        @no_op_if_value_is_null
        def path_to_bytes(path):
            with xopen(path, "rb") as f:
                bytes_ = f.read()
            return bytes_

        bytes_array = pa.array(
            [
                (path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"])
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                os.path.basename(path) if path is not None else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays(
            [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
        )
        return array_cast(storage, self.pa_type)


BACKBONE_ATOMS = ["N", "CA", "C", "O"]


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


@dataclass
class BackboneAtomArray:
    """BackboneAtomArray [`Feature`] to read protein backbone structure data from a PDB file.

    TODO: support other formats

    Input: The AtomArray feature accepts as input:
    - A `str`: Absolute path to the structure file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the structure file to the archive file.
        - `bytes`: Bytes of the structure file.

      This is useful for archived files with sequential access.
    - A `biotite.AtomArray` object.
    - A dictionary with the keys:
        - sequence
        - coords: an array of shape (N, 4, 3) or a dictionary with keys:
            - N
            - CA
            - C
            - O
    """

    def encode_example(self, value: Union[str, bytes, biotite.AtomArray]) -> dict:
        """Encode example into a format for Arrow.

        For now we encode the pdb str.
        TODO: we could potentially strip out all non-backbone atoms before encoding to save space...

        Args:
            value (`str`, `biotite.AtomArray`):
                Data passed as input to AtomArray feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if isinstance(value, Dict):
            atoms = []
            if "seq" in value:
                sequence = value["seq"]
            elif "sequence" in value:
                sequence = value["sequence"]
            else:
                raise ValueError("Sequence not found")
            if "coords" in value:
                coords = value["coords"]
                if isinstance(coords, list):
                    coords = np.array(coords)
                if isinstance(coords, np.ndarray):
                    assert coords.shape == (
                        len(sequence),
                        4,
                        3,
                    ), "Coordinates shape must be (N, 4, 3)"
                    res_coords = [
                        {
                            "N": res_coords[0],
                            "CA": res_coords[1],
                            "C": res_coords[2],
                            "O": res_coords[3],
                        }
                        for res_coords in coords
                    ]
                elif isinstance(coords, dict):
                    assert set(coords.keys()) == set(
                        BACKBONE_ATOMS
                    ), "Coordinate dict must contain all backbone atoms"
                    assert (
                        len(coords["N"])
                        == len(coords["CA"])
                        == len(coords["C"])
                        == len(coords["O"])
                        == len(sequence)
                    ), "All backbone coordinates must have the same length"
                    res_coords = []
                    for res_ix in range(len(sequence)):
                        res_coords.append({k: v[res_ix] for k, v in coords.items()})
                else:
                    raise ValueError(f"Unsupported coordinate type: {type(coords)}")
            elif "N" in value and "CA" in value and "C" in value and "O" in value:
                assert (
                    len(value["N"])
                    == len(value["CA"])
                    == len(value["C"])
                    == len(value["O"])
                    == len(sequence)
                ), "All backbone coordinates must have the same length"
                res_coords = []
                for res_ix in range(len(sequence)):
                    res_coords.append({k: v[res_ix] for k, v in value.items()})
            else:
                raise ValueError("Coordinates not found")

            atoms = []
            for res_ix, (aa, res_coords) in enumerate(zip(sequence, res_coords)):
                res_name = ProteinSequence.convert_letter_1to3(aa)
                for atom_ix, atom_name in enumerate(BACKBONE_ATOMS):
                    atom = bs.Atom(
                        coord=res_coords[atom_ix],
                        chain_id="A",
                        res_id=res_ix + 1,
                        res_name=res_name,
                        hetero=False,
                        atom_name=atom_name,
                        element=atom_name[0],
                    )
                    atoms.append(atom)
            value = bs.array(atoms)
        elif isinstance(value, bs.AtomArray):
            value = filter_backbone(value)
        return super().encode_example(value)

    def decode_example(self, value: dict, token_per_repo_id=None) -> "bs.AtomArray":
        """Decode example structure file into AtomArray data.

        Args:
            value (`str` or `dict`):
                A string with the absolute structure file path, a dictionary with
                keys:

                - `path`: String with absolute or relative structure file path.
                - `bytes`: The bytes of the structure file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                structure files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `biotite.AtomArray`
        """
        atom_array = super().decode_example(value, token_per_repo_id)
        return filter_backbone(atom_array)
