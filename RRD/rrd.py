# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:20:17 2022

@author: wanxiang.shen@u.nus.edu
"""


import pandas as pd
import numpy as np

from RRD.qm import get_qmdesc
from RRD.sterics import get_SHI, get_TSEI, get_buried_vol 
from RRD.bv_model import GetBV
from RRD.bde import get_bde
from RRD.bond import get_bond_basic_attrs
from RRD.pchem import get_polarizability, get_GasteigerCharge, get_MR

from joblib import Parallel, delayed
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

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
    dfb = dfb[dfb.columns[4:]]
    
    ## atom attrs
    gc = get_GasteigerCharge(smiles)
    mr = get_MR(smiles)
    apol = get_polarizability(smiles)
    shi = get_SHI(smiles)
    tsei = get_TSEI(smiles)
    
    dfa = dfa.join(gc).join(mr).join(apol).join(shi).join(tsei) #.join(dfv)
    
    #buried_volume
    dfv = pd.DataFrame(GetBV([smiles])[0], columns = ['PBV','PSV'])
    dfa = dfa.join(dfv)
        
    return dfa, dfb


atom_feature_config = {'maxv': {'PC': 0.6302622556686401,
                        'FN': 0.7218729853630066,
                        'FE': 0.5251923203468323,
                        'NSC': 500,
                        'GC': 0.6,
                        'MR': 14.02,
                        'APol': 10.6,
                        'SHI': 3.2216382711835254,
                        'TSEI': 2.8153517921163136,
                        'OV':200.0,
                        'PBV':100.0,
                        'PSV':100.0},
                       'minv': {'PC': -0.4310302436351776,
                        'FN': -0.061693187803030014,
                        'FE': -0.04397563263773918,
                        'NSC': -393.8583068847656,
                        'GC': -0.8569237321720045,
                        'MR': 0.0,
                        'APol': 0.557,
                        'SHI': 0.16777466065013924,
                        'TSEI': 0.0,
                        'OV':0.0,
                        'PBV':0.0,
                        'PSV':0.0}
                      }

bond_feature_config = {'maxv':{'SB': 1.0,
                     'DB': 1.0,
                     'TB': 1.0,
                     'AB': 1.0,
                     'CB': 1.0,
                     'RB': 1.0,
                     'SN': 1.0,
                     'SA': 1.0,
                     'SS': 1.0,
                     'SR': 1.0,
                     'BO': 2.9902005,
                     'BL': 2.2826374,
                     'BDE': 200,
                     'BDFE': 200},
                     'minv': {'SB': 0.0,
                     'DB': 0.0,
                     'TB': 0.0,
                     'AB': 0.0,
                     'CB': 0.0,
                     'RB': 0.0,
                     'SN': 0.0,
                     'SA': 0.0,
                     'SS': 0.0,
                     'SR': 0.0,
                     'BO': 0.5111524,
                     'BL': 0.9562112,
                     'BDE': 35.24322,
                     'BDFE': 19.995909}
                    }

class RRDCalculator:
    
    def __init__(self, scale=True):
        self.scale = scale
    
    def transform(self, smiles):
        '''
        smiles: smile string
        '''
        try:
            dfa, dfb = get_rrd(smiles)
            
        except:
            print('error when calculating %s' % smiles)
            dfa = pd.DataFrame([], columns  = atom_feature_config['maxv'].keys())
            dfb = pd.DataFrame([], columns  = bond_feature_config['maxv'].keys())
            
        if self.scale:
            a_scale = pd.DataFrame(atom_feature_config)
            a_scale['gap'] = a_scale.maxv - a_scale.minv
            dfa_scaled = (dfa - a_scale.minv) / a_scale.gap
            dfa_scaled = dfa_scaled[dfa.columns]
            dfa = dfa_scaled.fillna(0).clip(0,1)

            b_scale = pd.DataFrame(bond_feature_config)
            b_scale['gap'] = b_scale.maxv - b_scale.minv
            dfb_scaled = (dfb[b_scale.index] - b_scale.minv) / b_scale.gap
            dfb_scaled.BDE = dfb_scaled.BDE.fillna(1)
            dfb_scaled.BDFE = dfb_scaled.BDFE.fillna(1)
            dfb = dfb_scaled.fillna(0).clip(0,1)
           
        return dfa, dfb
    
    
    def batch_transform(self, smiles_list, n_jobs = -1):
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self.transform)(smiles,) for smiles in tqdm(smiles_list, ascii=True))
        return res