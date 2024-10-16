from typing import List

import biotite.structure as bs

from .protein import Protein


class ProteinComplex(Protein):
    """A protein complex."""

    def __init__(self, proteins: List[Protein]):
        self._chain_ids = [prot.chain_id for prot in proteins]
        self._proteins_lookup = {prot.chain_id: prot for prot in proteins}
        self.atoms = sum([prot.atoms for prot in proteins], bs.AtomArray())

    @property
    def get_chain(self, chain_id: str) -> Protein:
        return self._proteins_lookup[chain_id]

    def interface(
        self, atom_name: str = "CA", threshold: float = 8.0
    ) -> "ProteinComplex":
        raise NotImplementedError("Not implemented yet")
