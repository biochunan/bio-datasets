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
from typing import Any, ClassVar, Optional, Union, List

import biotite
import numpy as np
import pyarrow as pa
from biotite import structure as bs
from biotite.sequence import ProteinSequence
from biotite.structure import filter_amino_acids, get_chains
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile
from datasets import config
from datasets.download.download_config import DownloadConfig
from datasets.utils.file_utils import is_local_path, xopen, xsplitext
from datasets.utils.py_utils import string_to_dict
from datasets.features.features import register_feature
from bio_datasets import config as bio_config
from bio_datasets import constants as bio_constants
from .features import AtomArrayExtensionType


def atom_array_from_dict(d: dict) -> bs.AtomArray:
    sequence = d["sequence"]
    annots_keys = [k for k in d.keys() if k in AtomArrayExtensionType.extra_annots]
    if "backbone_coords" in d:
        assert len(sequence) == len(d["backbone_coords"]["N"])
        atoms = []
        for res_ix, (aa, res_coords) in enumerate(
            zip(sequence, d["backbone_coords"])
        ):
            res_name = ProteinSequence.convert_letter_1to3(aa)
            for atom_ix, atom_name in enumerate(bio_constants.BACKBONE_ATOMS):
                annots = {}
                for k in annots_keys:
                    annots[k] = d[k][res_ix]
                atom = bs.Atom(
                    coord=res_coords[atom_ix],
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
            model=model,
            assembly_id=assembly_id,
            extra_fields=extra_fields,
            chain_ids=chain_ids,
        )
    elif format == "pdb":
        pdbf = PDBFile.read(fpath_or_handler)
        structure = pdbf.get_structure(
            pdbf,
            model=model,
            assembly_id=assembly_id,
            extra_fields=extra_fields,
            chain_ids=chain_ids,
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
            assembly_id=assembly_id,
            extra_fields=extra_fields,
            chain_ids=chain_ids,
        )
    else:
        raise ValueError(f"Unsupported file format: {format}")

    return structure


class _AtomArrayFeature:
    def __init__(
        self,
        chain_ids: Optional[Union[str, List[str]]] = None,
        assembly_id: Optional[int] = None,
        with_occupancy: bool = False,
        with_b_factor: bool = False,
        with_atom_id: bool = False,
        with_charge: bool = False,
        with_element: bool = False,
    ):
        self.chain_ids = chain_ids
        self.assembly_id = assembly_id
        self.with_occupancy = with_occupancy
        self.with_b_factor = with_b_factor
        self.with_atom_id = with_atom_id
        self.with_charge = with_charge
        self.with_element = with_element


@dataclass
class AtomArray(_AtomArrayFeature):
    """AtomArray [`Feature`] to read macromolecular atomic structure data from a PDB or CIF file.

    This feature stores the array directly as a pa struct (basically a dictionary of arrays),
    as defined in the AtomArrayExtensionType.
    """
    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="AtomArray", init=False, repr=False)   # probably requires registered feature type

    def __call__(self):
        pa_type = AtomArrayExtensionType(self.with_occupancy, self.with_b_factor, self.with_atom_id, self.with_charge, self.with_element)
        return pa_type


register_feature(AtomArray, "AtomArray")


@dataclass
class BiomolecularStructureFile(_AtomArrayFeature):
    """BiomolecularStructureFile [`Feature`] to read macromolecular atomic structure data from supported file types.
    The file contents is serialized as bytes or a file path (or both) within an Arrow table.
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
    - A `biotite.AtomArray` object.
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

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the structure data. If `False`,
            returns the underlying dictionary in the format `{"path": structure_path, "bytes": structure_bytes}`.
    """

    decode: bool = True
    id: Optional[str] = None
    encode_with_foldcomp: bool = False
    foldcomp_anchor_residue_threshold: int = (
        25  # < 0.1 A RMSD (less than noise typically applied in generative models)
    )
    mode: str = "array"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="AtomArray", init=False, repr=False)

    def __post_init__(self):
        # q: what does dtype actually do?
        if self.mode == "array":
            self.dtype = "bs.AtomArrray"
        elif self.mode == "structure":
            raise NotImplementedError()  # TODO use Biopython

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

        This determines what gets written to the Arrow file.
        """

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, bytes):
            return {"path": None, "bytes": value}
        elif isinstance(value, dict):
            value = atom_array_from_dict(value)
            return self.encode_example(value)
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


register_feature(BiomolecularStructureFile, "BiomolecularStructureFile")


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

    return filter_atom_names(array, bio_constants.BACKBONE_ATOMS) & filter_amino_acids(array)


@dataclass
class ProteinBackboneStructureFile(BiomolecularStructureFile):
    """BackboneAtomArray [`Feature`] to read/store protein backbone structure data from a PDB file.
    Basically a convenience class to support lower-memory storage of protein backbone data.

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
        if isinstance(value, dict):
            value = atom_array_from_dict(value)
            value = filter_backbone(value)
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


register_feature(ProteinBackboneStructureFile, "ProteinBackboneStructureFile")
