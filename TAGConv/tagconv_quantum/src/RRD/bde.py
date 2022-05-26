# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:29:54 2022

@author: wanxiang.shen@u.nus.edu
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from alfabet import model as bde_model

def get_bde(smiles):
    '''
     homolytic bond dissociation energies (BDEs)
    '''
    cols = ['bde_pred', 'bdfe_pred'] #, 
    df = bde_model.predict([smiles], drop_duplicates = False)
    df_bde_res = df.set_index('bond_index')[cols]
    
    edge_indices = []
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetIdx(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                          bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]]
    
    df1 = pd.DataFrame(edge_indices, columns= [ 'BondIdx', 'BeginAtomIdx', 
                                             'EndAtomIdx','BeginAtomSymbol', 
                                             'EndAtomSymbol'])
    
    df1 = df1.set_index('BondIdx')
    
    
    dfbde = df1.join(df_bde_res)
    
    ## Fill NaN based on equivalent hydrogen
    fillna_bde = dfbde.groupby(['BeginAtomIdx', 
                                'BeginAtomSymbol', 
                                'EndAtomSymbol'])[cols].apply(lambda x: x.fillna(method='ffill'))
    
    dfbde[cols] = fillna_bde
    
    dfbde = dfbde.rename(columns={'bde_pred':'BDE', 'bdfe_pred':'BDFE'})
    return dfbde
