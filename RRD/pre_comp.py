import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm
# from rrd import RRDCalculator
# RRDC = RRDCalculator(scale=True)

def removemap(rxnstr):
    """
    Remove atom-mapping
    """
    rxn = rdChemReactions.ReactionFromSmarts(rxnstr, useSmiles=True)
    rdChemReactions.RemoveMappingNumbersFromReactions(rxn)
    return rdChemReactions.ReactionToSmiles(rxn)

df = pd.read_csv('../data/cleaned_uspto50k.csv')
rxn_list = df['rxn_smiles'].drop_duplicates().values.tolist()
rxn_list = [removemap(rxn) for rxn in tqdm(rxn_list)]
# print(rxn_list)
smiles_list = []
for rxn in tqdm(rxn_list):
    smiles = rxn.replace('>', '.').split('.')
    # print(smiles)
    smiles_list += [e for e in smiles if not e == '']

print(len(smiles_list))
smiles_list = list(set(smiles_list))
print(len(smiles_list))