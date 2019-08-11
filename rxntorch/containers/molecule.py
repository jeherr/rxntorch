from __future__ import print_function

import rdkit.Chem as Chem


class Mol(object):
    """
    """
    def __init__(self, smile):
        self.smile = smile

    def canonicalize(self):
        self.smile = Chem.MolToSmile(Chem.MolFromSmile(self.smile))