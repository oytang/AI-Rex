# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:20:17 2022

@author: wanxiang.shen@u.nus.edu
"""


import pandas as pd
import numpy as np
from rdkit import Chem


def get_bond_basic_attrs(smiles):
    
    #smiles = r"O[C@H]1C2CC3C[C@@H](C2)CC1C3"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    '''
    Bond attributes calculation, 10-dim
    1) Single
    2) Double
    3) Triple
    4) Aromatic
    5) Conjugation
    6) Ring
    7) STEREONONE
    8) STEREOANY
    9) STEREOZ (S)
    10) STEREOE (R)
    '''
    
    stereos = [Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOANY,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREOE,]

    edge_indices = []
    edge_attrs = []
    
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetIdx(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                          bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]]
        
        bond_type = bond.GetBondType()
        single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
        double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
        triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
        aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
        conjugation = 1. if bond.GetIsConjugated() else 0.
        ring = 1. if bond.IsInRing() else 0.
        stereo = [0.] * 4
        stereo[stereos.index(bond.GetStereo())] = 1.

        edge_attr = [single, double, triple, aromatic, conjugation, ring] + stereo

        ## commented out code
        edge_attrs += [edge_attr] #, edge_attr
        # edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(),
        #                   bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol()]]
        
    edge_index = np.array(edge_indices)
    edge_attrs = np.array(edge_attrs)
    edge_attr_names = np.array(['SB', 'DB', 'TB', 
                                'AB', 'CB', 'RB',
                                'SN', 'SA', 'SS', 'SR'])
    
    df1 = pd.DataFrame(edge_index, columns= [ 'BondIdx', 'BeginAtomIdx', 'EndAtomIdx','BeginAtomSymbol', 'EndAtomSymbol'])
    df2 = pd.DataFrame(edge_attrs, columns = edge_attr_names)
 
    dfb = df1.join(df2)
    
    dfb = dfb.set_index('BondIdx')
    dfb.index = dfb.index.astype(int)
    
    return dfb