from tqdm import tqdm

from collections import Counter
from multiprocessing import Pool
from rxntorch.containers.dataset import RxnDataset as rxnd

dataset = rxnd("2001_Sep2016_USPTO_applications_smiles_canonical.dat", path="data/USPTO/")
n = len(dataset)
pbar=tqdm(total=n)
result = [None]*n
rxn_smiles = dataset.rxn_smiles


def build_rxn_vocab(i):
    smile = rxn_smiles[i]
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
    return i, symbols


def update(i, symbols):
    result[i] = symbols
    pbar.update()

pool = Pool(processes=8)
for i in range(n):
    pool.apply_async(build_rxn_vocab, args=(i,), callback=update)
pool.close()
pool.join()
pbar.close()
#counter = result[0]
#for i in range(1, 8):
#    counter.update(result[i])

