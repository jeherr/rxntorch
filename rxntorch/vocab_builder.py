import tqdm

from collections import Counter
import multiprocessing as mp
from rxntorch.containers.dataset import RxnDataset as rxnd
from rxntorch import smiles_parser as sp


def build_rxn_vocab(smile):
    try:
        symbols = []
        reactants, reagents, products = smile.split('>')
        reactants, reagents, products = reactants.split('.'), reagents.split('.'), products.split('.')

        for reactant in reactants:
            symbols += sp.parser_list(reactant)
        for reagent in reagents:
            symbols += sp.parser_list(reagent)
        for product in products:
            symbols += sp.parser_list(product)
        symbols += ['.'] * (len(reactants) + len(reagents) + len(products) - 3)
        symbols += ['>']
        return symbols
    except:
       return smile 


if __name__ == '__main__':
	dataset = rxnd("2001_Sep2016_USPTO_applications_smiles_canonical.dat", path="data/USPTO/")
	n = len(dataset)
	rxn_smiles = dataset.rxn_smiles
	pool = mp.Pool(mp.cpu_count())
	result = list(tqdm.tqdm(pool.imap(build_rxn_vocab, rxn_smiles), total=n))
	pool.close()
	pool.join()
	counter = Counter()
	for item in result:
		if type(item) == str:
			continue
		for symbol in item:
			counter[symbol] += 1
	print(counter['C'])

