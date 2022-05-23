from cProfile import label
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from tqdm import tqdm

from falsify_data import removemap
from utils import *

import argparse

# argument parsing
parser = argparse.ArgumentParser()

# number of random false reactions to sample
parser.add_argument('--num_false', type=int, default=42000)
parser.add_argument('--train_size', type=float, default=0.7)
parser.add_argument('--valid_size', type=float, default=0.2)
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--ECFP4nBits', type=int, default=1024)

args = parser.parse_args()

assert np.isclose(args.train_size + args.valid_size + args.test_size, 1.0), 'train_size, valid_size and test_size should sum to 1.0!!!'

# handle postive rxns
pos_rxn = pd.read_csv('cleaned_uspto50k.csv')
pos_rxn['label'] = 1
# preprocess to let true dataset look more like false dataset
print('remove atom map')
pos_rxn['rxn_smiles'] = [removemap(rxn) for rxn in tqdm(pos_rxn['rxn_smiles'])]
pos_rxn_can = []
# split reactants & products
print('clean reaction SMILES format')
for rxn in tqdm(pos_rxn['rxn_smiles']):
    rt, pt = rxn2rtpt(rxn)
    # canonicalize products SMILES
    pt = can_SMILES(pt)
    # canonicalize reaction SMILES format
    pos_rxn_can.append('>>'.join((rt, pt)))
pos_rxn['rxn_smiles'] = pos_rxn_can
pos_rxn = pos_rxn[['rxn_smiles', 'label']]
pos_rxn = pos_rxn.drop_duplicates(subset='rxn_smiles').reset_index(drop=True)


# handle negative rxns
neg_rxn_1 = pd.read_csv('negative_strict.csv')
neg_rxn_1 = neg_rxn_1[['rxn_smiles', 'label']]
neg_rxn_1 = neg_rxn_1.drop_duplicates(subset='rxn_smiles')
neg_rxn_2 = pd.read_csv('negative_random_comprehensive.csv')
neg_rxn_2 = neg_rxn_2.drop_duplicates(subset='rxn_smiles').reset_index(drop=True)
neg_rxn_2 = neg_rxn_2[['rxn_smiles', 'label']]
# random sub-sample neg_rxn_2 to have a equally labeled dataset
neg_rxn_2 = neg_rxn_2.loc[np.random.choice(a=len(neg_rxn_2), size=args.num_false, replace=False)]
neg_rxn = pd.concat([neg_rxn_1, neg_rxn_2]).drop_duplicates(subset='rxn_smiles').reset_index(drop=True)


# integrate true and false rxns
all_rxn = pd.concat([pos_rxn, neg_rxn])
# suffle rows & reset index
all_rxn = all_rxn.drop_duplicates(subset='rxn_smiles').sample(frac=1).reset_index(drop=True)
rt_list = []
for rxn in all_rxn['rxn_smiles']:
    rt, _ = rxn2rtpt(rxn)
    # sort reactants in a uniform order
    rt_list.append('.'.join(sorted(rt.split('.'))))
all_rxn['rt'] = rt_list
print('generate ECFP4')
all_rxn['ECFP4'] = [list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=args.ECFP4nBits)) for smi in tqdm(rt_list)]


# conditional split of dataset
X = np.array(all_rxn['ECFP4'].values.tolist())
# the first PC grabs major variance
pca = PCA(n_components=1)
pca.fit(X.transpose())
X_pca = pca.components_.transpose()


# Two-tail double-side split
# calculate percentile from split size
train_prec = [
    [(1-args.train_size)*0.25, 0.25+args.train_size*0.25], 
    [0.5+(1-args.train_size)*0.25, 0.75+args.train_size*0.25]
]
valid_prec = [
    [args.test_size*0.25, (args.test_size+args.valid_size)*0.25], 
    [0.25+args.train_size*0.25, 0.25+(args.train_size+args.valid_size)*0.25], 
    [0.5+args.test_size*0.25, 0.5+(args.test_size+args.valid_size)*0.25], 
    [0.75+args.train_size*0.25, 0.75+(args.train_size+args.valid_size)*0.25]
]
test_prec = [
    [0.0, args.test_size*0.25],
    [0.25+(args.train_size+args.valid_size)*0.25, 0.5],
    [0.5, 0.5+args.test_size*0.25],
    [0.75+(args.train_size+args.valid_size)*0.25, 1.0]
]

train_split = [np.percentile(X_pca[:,0], np.array(itv)*100) for itv in train_prec]
valid_split = [np.percentile(X_pca[:,0], np.array(itv)*100) for itv in valid_prec]
test_split = [np.percentile(X_pca[:,0], np.array(itv)*100) for itv in test_prec]
# ensure inclusion of tails
test_split[0][0] -= 1
test_split[3][1] += 1

def getIDFromSplit(X, split):
    ids = []
    for itv in split:
        upBound = np.arange(len(X))[X <= itv[1]]
        loBound = np.arange(len(X))[X > itv[0]]
        ids += list(set(upBound).intersection(set(loBound)))
    return ids

train_id = getIDFromSplit(X_pca[:,0], train_split)
valid_id = getIDFromSplit(X_pca[:,0], valid_split)
test_id = getIDFromSplit(X_pca[:,0], test_split)


# export splitted dataset
all_rxn = all_rxn[['rxn_smiles', 'label']]
all_rxn.loc[train_id].to_csv('data_train.csv', index=False)
all_rxn.loc[valid_id].to_csv('data_valid.csv', index=False)
all_rxn.loc[test_id].to_csv('data_test.csv', index=False)