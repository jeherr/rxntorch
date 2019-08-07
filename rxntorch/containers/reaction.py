from __future__ import print_function

from rxntorch.molecule import Mol

class Rxn(object):
    """
    """
    def __init__(self, smiles, bond_changes):
        self.smiles = smiles
        self.bond_changes = bond_changes
        self.reacts, self.prods = self.split_reacts_prods()

    def split_reacts_prods(self):
        """Splits the reaction string into reactants and products, then
        further splits each substring into molecules. Molecules are then
        read by their SMILES string into rdkit Mol objects.

        Returns:
            reacts (list): list of rdkit Mols on the reactants side
            prods  (list): list of rdkit Mols on the products side
        """
        react_str, prod_str = self.smiles.split('>')
        reacts_strs = react_str.split('.')
        prods_strs = prod_str.split('.')
        reacts = [Mol(reacts_str) for reacts_str in reacts_strs]
        prods = [Mol(prods_str) for prods_str in prods_strs]
        return reacts, prods

    def smiles_to_graph(self):
