"""
Arrow ExtensionType for biotite AtomArray.
This type allows us to save AtomArrays as fields in arrow tables.
"""
import json

import pyarrow as pa

from datasets.features.features import string_to_arrow


class AtomArrayExtensionType(pa.ExtensionType):

    """AtomArray attributes:
    
    ----------
    {annot} : ndarray
        Multiple n-length annotation arrays.
    coord : ndarray, dtype=float, shape=(n,3)
        ndarray containing the x, y and z coordinate of the
        atoms.
    bonds : BondList or None
        A :class:`BondList`, specifying the indices of atoms
        that form a chemical bond.
    box : ndarray, dtype=float, shape=(3,3) or None
        The surrounding box. May represent a MD simulation box
        or a crystallographic unit cell.
    shape : tuple of int
        Shape of the atom array.
        The single value in the tuple is
        the length of the atom array.

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
    annots = [
        "chain_id",
        "res_id",
        "ins_code",
        "res_name",
        "hetero",
        "atom_name",
    ]
    extra_annots = [
        "b_factor",
        "occupancy",
        "charge",
        "element",  # seems redundant
    ]

    def __init__(
        self,
        with_bonds: bool,
        with_box: bool,
        with_occupancy: bool,
        with_b_factor: bool,
        with_charge: bool,
        with_element: bool,
        coords_dtype: str = "float32",
    ):
        self.with_box = with_box
        self.with_bonds = with_bonds
        self.with_occupancy = with_occupancy
        self.with_b_factor = with_b_factor
        self.with_charge = with_charge
        self.with_element = with_element
        self.coords_dtype = coords_dtype
        fields = [
            pa.field("coords", self._generate_array_dtype(coords_dtype, (None, 3))),  # 2D array with shape (None, 3)
            pa.field("res_id", pa.list_(pa.int32())),           # position in chain
            pa.field("res_name", pa.list_(pa.utf8())),          # residue name
            pa.field("atom_name", pa.list_(pa.utf8())),         # CA, C, N, etc.
            pa.field("chain_id", pa.list_(pa.utf8()), nullable=True),          # dtype="U4"
            pa.field("ins_code", pa.list_(pa.utf8()), nullable=True),          # dtype="U1"
            pa.field("hetero", pa.list_(pa.bool_()), nullable=True),           # dtype=bool
        ]
        if with_box:
            fields.append(pa.field("box", pa.list_(pa.list_(pa.float32(), 3), 3), nullable=True))
        if with_bonds:
            fields.append(
                pa.field(
                    "bonds", 
                    pa.list_(
                        pa.struct(
                            fields=[
                                pa.field("atom1_idx", pa.int32()), 
                                pa.field("atom2_idx", pa.int32()), 
                                pa.field("bond_type", pa.int8())
                            ]
                        )
                    ),
                    nullable=True,
                ),
            )
        if with_occupancy:
            fields.append(pa.field("occupancy", pa.list_(pa.float32()), nullable=True))
        if with_b_factor:
            fields.append(pa.field("b_factor", pa.list_(pa.float32()), nullable=True    ))
        if with_charge:
            fields.append(pa.field("charge", pa.list_(pa.int32()), nullable=True))
        if with_element:
            fields.append(pa.field("element", pa.list_(pa.utf8()), nullable=True))
        struct_type = pa.struct(fields)
        pa.ExtensionType.__init__(self, struct_type, f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __arrow_ext_serialize__(self):
        return json.dumps(
            (
                self.with_box,
                self.with_bonds,
                self.with_occupancy,
                self.with_b_factor,
                self.with_charge,
                self.coords_dtype,
            )
        ).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        args = json.loads(serialized)
        return cls(*args)

    def __reduce__(self):
        return self.__arrow_ext_deserialize__, (self.storage_type, self.__arrow_ext_serialize__())

    def __hash__(self):
        return hash(
            (
                self.__class__,
                self.with_bonds,
                self.with_box,
                self.with_occupancy,
                self.with_b_factor,
                self.with_charge,
                self.coords_dtype,
            )
        )

    def _generate_array_dtype(self, dtype, shape):
        # source: datasets ArrayXDExtensionType
        dtype = string_to_arrow(dtype)
        for d in reversed(shape):
            dtype = pa.list_(dtype)
            # Don't specify the size of the list, since fixed length list arrays have issues
            # being validated after slicing in pyarrow 0.17.1
        return dtype


pa.register_extension_type(AtomArrayExtensionType)
