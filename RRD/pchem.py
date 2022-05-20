# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:20:17 2022

@author: wanxiang.shen@u.nus.edu
"""


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors


############################################################
# Assign polarizabilities to every atom.
# Units are Angstrom^3/2. We assume all valences are filled.
#
# This will work without explicit hydrogens, but it will
# obviously not assign properties to implicit hydrogens.
#
# Values are from Table I of
# Miller and Savchik, JACS 101(24) 7206-7213, 1979.
# dx.doi.org/10.1021/ja00518a014
############################################################
def get_polarizability(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        #get degree for specific molecule
        nBonds = atom.GetTotalDegree()

        #Hydrogen (H)
        if atom.GetAtomicNum() == 1:
            atom.SetDoubleProp('polarizability', 0.314)

        #Carbon (C)
        elif atom.GetAtomicNum() == 6:
            if nBonds == 4:
                atom.SetDoubleProp('polarizability', 1.294)
            elif nBonds == 2:
                atom.SetDoubleProp('polarizability', 1.393)
            elif nBonds == 3:
                if atom.GetNumExplicitHs() + atom.GetNumImplicitHs() > 0:
                    atom.SetDoubleProp('polarizability', 1.428)
                else:
                    '''
                    in this part I employ a different logic than that in
                    previous version of ACSESS and may be consult with Aaron for
                    further details
                    '''
                    cross = True
                    for nbor in atom.GetNeighbors():
                        if not nbor.GetIsAromatic():
                            cross = False
                            break
                    if cross:
                        atom.SetDoubleProp('polarizability', 1.800)
                    else:
                        atom.SetDoubleProp('polarizability', 1.428)

        #Nitrogen (N)
        elif atom.GetAtomicNum() == 7:
            if atom.GetIsAromatic():
                if nBonds == 2:
                    atom.SetDoubleProp('polarizability', 1.262)
                else:
                    atom.SetDoubleProp('polarizability', 1.220)
            else:
                if nBonds == 1:
                    atom.SetDoubleProp('polarizability', 1.304)
                else:
                    atom.SetDoubleProp('polarizability', 1.435)

        #Oxygen (O)
        elif atom.GetAtomicNum() == 8:
            if atom.GetIsAromatic():
                atom.SetDoubleProp('polarizability', 1.099)
            else:
                if nBonds == 1:
                    atom.SetDoubleProp('polarizability', 1.216)
                else:
                    atom.SetDoubleProp('polarizability', 1.290)

        #Sulfur (S)
        elif atom.GetAtomicNum() == 16:
            if atom.IsAromatic():
                atom.SetDoubleProp('polarizability', 2.982)
            elif nBonds == 2:
                atom.SetDoubleProp('polarizability', 3.496)
            else:
                atom.SetDoubleProp('polarizability', 3.967)

        #Halogens
        elif atom.GetAtomicNum() == 9:
            atom.SetDoubleProp('polarizability', 1.046)
        elif atom.GetAtomicNum() == 15:
            atom.SetDoubleProp('polarizability', 3.000)
        elif atom.GetAtomicNum() == 17:
            atom.SetDoubleProp('polarizability', 3.130)
        elif atom.GetAtomicNum() == 35:
            atom.SetDoubleProp('polarizability', 5.577)
        elif atom.GetAtomicNum() == 53:
            atom.SetDoubleProp('polarizability', 8.820)

        #Iridium (I)
        #This param value was obtained by fitting the above known tau values
        #In general polarizability increases with atomic number so we used
        #linear fit to get the value
        #This is a crudest approx so could be wrong!
        elif atom.GetAtomicNum() == 77:
            atom.SetData('polarizability', 12.77)

        else:
            raise KeyError('No polarizabilities for atomic number' +
                           str(atom.GetAtomicNum()))
    res = []
    for at in mol.GetAtoms():
        res.append([at.GetIdx(), at.GetDoubleProp("polarizability")])
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
