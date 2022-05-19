# Basic workflow credit to qianlong's script
# The codes are comestically decorated here
# with some necessary bug fix

import pandas as pd
import rdchiral.main as rdc
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm, trange

from falsify_data import removemap, clear_atom_map


def reverse_temp(temp):
    """
    Convert retro template to forward template
    This was re-implemented for
    (1) handle the exception where there are more than one molecule as product
    (2) better readability
    """
    pt, rts = temp.split(">>")

    return rts.replace(').(', '.') + '>>' + pt.replace(').(', '.')


def rxn2rtpt(rxn_smi):
    """
    Convert reaction SMILES into reactants and products
    """
    # for normal reactions
    if rxn_smi.count('>>') == 1:
        return rxn_smi.split('>>')

    # for reactions with one reactant in middle
    # like 'Cc1ccc(C(=O)O)cc1F>O=C1CCC(=O)N1Br>O=C(O)c1ccc(CBr)c(F)c1'
    elif rxn_smi.count('>') == 2:
        rt1, rt2, pt = rxn_smi.split('>')
        # there are possible cases without rt1
        # like '>O=C(O)C1CCN(C(=O)CO)CC1>O'
        if rt1 == '':
            return rt2, pt
        else:
            return f'{rt1}.{rt2}', pt
    
    # Unknown situation, might happen in uncleaned dataset
    raise ValueError(f'Reaction SMILES is not valid: {rxn_smi}')