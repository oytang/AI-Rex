# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:20:17 2022

@author: wanxiang.shen@u.nus.edu
"""


import pandas as pd
import numpy as np

from qm import get_qmdesc
from sterics import get_buried_vol, get_SHI, get_TSEI
from bde import get_bde
from bond import get_bond_basic_attrs
from pchem import get_polarizability, get_GasteigerCharge, get_MR



def get_rrd(smiles):

    """
    # atom
    PC	Hirshfeld partial charge
    FN	Neucleuphilic Fukui indices
    FE	Electrophilic Fukui indices
    NSC	NMR shielding constants
    GC	GasteigerCharge
    MR	Molar refractivity
    Apol	Atomic polarizability
    SHI	Steric hindrance index
    TSEI	Toplogical steric effect index
    
    # bond
    SB	Single bond
    DB	Double bond
    TB	Triple bond
    AB	Aromatic bond
    CB	Conjugation bond
    RB	Ring bond
    SN	Stereo-None
    SA	Stereo-Any
    SS	Stereo-S
    SR	Stereo-R
    BO	Bond order
    BL	Bond length
    BDE	Bond dissociation enthalpies
    BDFE	Bond dissociation free energies


    """
    dfb = get_bond_basic_attrs(smiles)
    dfa, dfbb = get_qmdesc(smiles)
    df_bde = get_bde(smiles)

    ## bond attrs
    dfb = dfb.join(dfbb).join(df_bde[['BDE', 'BDFE']])

    ## atom attrs
    gc = get_GasteigerCharge(smiles)
    mr = get_MR(smiles)
    apol = get_polarizability(smiles)
    shi = get_SHI(smiles)
    tsei = get_TSEI(smiles)
    
    #buried_volume
    dfv = get_buried_vol(smiles)
    dfa = dfa.join(gc).join(mr).join(apol).join(shi).join(tsei).join(dfv)

    return dfa, dfb