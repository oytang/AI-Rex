# -*- coding: utf-8 -*-
# @Time    : 2022.05.12
# @Author  : qianlong
# @File    : falsify_data.py

import pandas as pd
import rdchiral.main as rdc
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm, trange

def removemap(rxnstr):
    """
    Remove atom-mapping
    """
    rxn = rdChemReactions.ReactionFromSmarts(rxnstr, useSmiles=True)
    rdChemReactions.RemoveMappingNumbersFromReactions(rxn)
    return rdChemReactions.ReactionToSmiles(rxn)

def reverse_temp(temp):
    """
    Convert retro template to forward template
    """
    pt, rts = temp.split(">>")
    rt_list = rts.split(".")
    new_rt_list = []
    for rt in rt_list:
        new_rt_list.append(rt[1:-1])
    print(new_rt_list)
    return "("+").(".join(new_rt_list)+")"+">>"+ pt[1:-1]

def clear_atom_map(smis):
    mol = Chem.MolFromSmiles(smis)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

def load_csv(path):

    X, y = [], []
    df = pd.read_csv(path, header=0)
    # df = df[:10]
    num = len(df)
    rxn_smiles = list(df['rxn_smiles'])
    retro_templates = list(df['retro_template'])


    for i in tqdm(range(num)):
        rxn = rxn_smiles[i]
        rxn = removemap(rxn)
        X.append(rxn)
        y.append(1)  # add correct rxn
        try:
            pt, rt = rxn.split(">>")
        except:
            continue
        rule = retro_templates[i]
        retro_outcomes = rdc.rdchiralRunText(rule, pt)
        # print("retro_outcomes", retro_outcomes)

        #######Method ONE STRICT#######
        for outcome in retro_outcomes:
            if rt == outcome:
                # print("Correct")
                continue
            else:
                bad_rxn = outcome + ">>" + pt
                X.append(bad_rxn)
                # print("Wrong ", rxn)
                y.append(0)

    #######Method TWO Random#######
    for _ in tqdm(range(1000)):
        import random
        random_temp_index = random.randint(0, num-1)
        print(random_temp_index)
        retro_template = df.iloc[random_temp_index]["retro_template"]
        for i in range(num):
            reactants = df.iloc[i]["reactants"]
            products = df.iloc[i]["products"]
            rule = reverse_temp(retro_template)
            retro_outcomes = rdc.rdchiralRunText(rule, clear_atom_map(reactants))
            #
            if len(retro_outcomes) == 0:
                continue
            if retro_outcomes[0] == clear_atom_map(products):
                continue
            bad_rxn = clear_atom_map(reactants)+">>"+retro_outcomes[0]
            if bad_rxn in X:
                continue

            X.append(bad_rxn)
            print("Wrong ", bad_rxn)
            y.append(0)

    data = {'reaction': X, 'label': y}
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    path = "./cleaned_uspto50k.csv"
    df = load_csv(path)
    df.to_csv("data.csv")
    print(df)