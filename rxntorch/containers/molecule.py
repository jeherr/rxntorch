from __future__ import print_function

import rdkit.Chem as Chem


class Mol(object):
    """
    """
    def __init__(self, smile):
        self.smile = smile
        self.canonical = False

    def canonicalize(self):
        self.smile = Chem.MolToSmiles(Chem.MolFromSmiles(self.smile))
        self.canonical = True