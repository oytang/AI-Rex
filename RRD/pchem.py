# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:20:17 2022

@author: wanxiang.shen@u.nus.edu
"""


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors



def get_polarizability(smiles,includeImplicitHs=True):
    # apol values from https://github.com/cdk/cdk/blob/master/descriptor/qsarmolecular/src/main/java/org/openscience/cdk/qsar/descriptors/molecular/APolDescriptor.java
    # the original source cited in that code is no longer available and there is no other reference
    atomPols = [0, 0.666793, 0.204956, 24.3, 5.6, 3.03, 1.76, 1.1, 0.802, 0.557, 0.3956,
                    23.6, 10.6, 6.8, 5.38, 3.63, 2.9, 2.18, 1.6411, 43.4, 22.8, 17.8, 14.6, 12.4, 11.6, 9.4, 8.4, 7.5,
                    6.8, 6.1, 7.1, 8.12, 6.07, 4.31, 3.77, 3.05, 2.4844, 47.3, 27.6, 22.7, 17.9, 15.7, 12.8, 11.4, 9.6,
                    8.6, 4.8, 7.2, 7.2, 10.2, 7.7, 6.6, 5.5, 5.35, 4.044, 59.6, 39.7, 31.1, 29.6, 28.2, 31.4, 30.1,
                    28.8, 27.7, 23.5, 25.5, 24.5, 23.6, 22.7, 21.8, 21, 21.9, 16.2, 13.1, 11.1, 9.7, 8.5, 7.6, 6.5,
                    5.8, 5.7, 7.6, 6.8, 7.4, 6.8, 6, 5.3, 48.7, 38.3, 32.1, 32.1, 25.4, 27.4, 24.8, 24.5, 23.3, 23,
                    22.7, 20.5, 19.7, 23.8, 18.2, 17.5]
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    res = []
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum<=len(atomPols):
            apol = atomPols[anum]
            if includeImplicitHs:
                apol += atomPols[1] * atom.GetTotalNumHs(includeNeighbors=False)
            res.append([atom.GetIdx(), apol])
        else:
            res.append([atom.GetIdx(), 0.])

    df_apol = pd.DataFrame(res, columns = ['AtomIdx', 'APol'])
    df_apol = df_apol.set_index('AtomIdx')
    
    return df_apol


def get_GasteigerCharge(smiles):
    """
    GasteigerCharge
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol)

    for at in mol.GetAtoms():
        at.GetDoubleProp("_GasteigerCharge")

    res = []
    for at in mol.GetAtoms():
        res.append([at.GetIdx(), at.GetDoubleProp("_GasteigerCharge")])
    df_GC = pd.DataFrame(res, columns = ['AtomIdx', 'GC'])
    df_GC = df_GC.set_index('AtomIdx')
    return df_GC


def get_MR(smiles):
    """
    the Wildman-Cripppen mr value
    """
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mrContribs = rdMolDescriptors._CalcCrippenContribs(mol)
    mrs = [y for x,y in mrContribs]
    df_MR = pd.DataFrame(mrs, columns=['MR'])
    df_MR.index.name = 'AtomIdx'
    return df_MR
