# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:20:17 2022

@author: wanxiang.shen@u.nus.edu
"""


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from qmdesc import ReactivityDescriptorHandler

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

RDH = ReactivityDescriptorHandler()

def get_qmdesc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)


    results = RDH.predict(smiles)

    s = pd.Series(results)
    HPC = s.partial_charge
    NFI = s.fukui_neu
    EFI = s.fukui_elec
    NSC = s.NMR
    BO = s.bond_order
    BL = s.bond_length

    atom_index = np.array([atom.GetIdx() for atom in mol.GetAtoms()])
    bond_index = np.array([bond.GetIdx() for bond in mol.GetBonds()])

    dfb = pd.DataFrame(np.stack([BO, BL], axis=1), index = bond_index, 
                       columns = ['BO', 'BL'])
    dfb.index.name = 'BondIdx'
    
    dfa = pd.DataFrame(np.stack([HPC, NFI, EFI, NSC], axis=1), index = atom_index, 
                       columns = ['PC', 'FN', 'FE', 'NSC'])
    dfa.index.name = 'AtomIdx'
    
    return dfa, dfb


