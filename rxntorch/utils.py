from __future__ import print_function

import random

def rxn_smiles_reader(txt_file):
    """Loads txt from a files containing reaction SMILES.

    Files should be in the format of one reaction string per line. Additional
    data can be added onto the end of the file in comma-seperated values. The
    order of data will need to be standardized.

    Args:
        txt_file (str): Path to the csv file containing the data

    Returns:
        bins (dict): Dictionary of binned reaction strings by size. Keys
            are the bin size and values are lists of the reaction strings.
    """
    bin_size = [10,20,30,40,50,60,80,100,120,150]
    bins = [[] for i in range(len(bin_size))]
    with open(txt_file, "r") as datafile:
        for line in datafile:
            r,e = line.strip("\r\n ").split()
            c = count(r)
            for i, size in enumerate(bin_size):
                if c <= size:
                    bins[i].append((r,e))
                    break

    for ibin in bins:
        random.shuffle(ibin)
    bins = {bin_size[i]: bins[i] for i in range(len(bin_size)) if (len(bins[i]) > 0)}
    return bins

def count(s):
    """Counts the number of heavy atoms in a reaction string."""
    c = 0
    for i in range(len(s)):
        if s[i] == ':':
            c += 1
    return c
