import os
import dgl
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from rdkit import Chem
import glob
import warnings

warnings.filterwarnings("ignore")


class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, mode, raw_graphs=None):
       
        self.args = args
        self.mode = mode
        self.raw_graphs = raw_graphs
        self.reactant_graphs = []
        self.product_graphs = []
        self.labels = []
        super().__init__(name='Smiles_' + mode)

    def process(self):
        print(f"transform {self.mode} data to dgl graphs")
        for i, (reactants_mol, reactants_fea, products_mol, products_fea, label) in tqdm(enumerate(self.raw_graphs), total=len(self.raw_graphs)):
            reactant_graph = mol_to_dgl(reactants_mol, reactants_fea)
            product_graph = mol_to_dgl(products_mol, products_fea)
            self.reactant_graphs.append(reactant_graph)
            self.product_graphs.append(product_graph)
            self.labels.append(torch.tensor(label).float())
        
    def __getitem__(self, i):
        return self.reactant_graphs[i], self.product_graphs[i], self.labels[i]

    def __len__(self):
        return len(self.reactant_graphs)


def mol_to_dgl(raw_graph, feature):
    # add edges
    #reactants_mol[mol, mol, mol], reactants_fea[(dfa, dfb), (dfa, dfb)]
    atoms = np.array([0] + [mol.GetNumAtoms() for mol in raw_graph])
    atom_cumsum = atoms.cumsum().tolist()[:-1]

    atom_fea = [df[0] for df in feature]
    bond_fea = [df[1] for df in feature]

    edges_src = [df.BeginAtomIdx.values.astype(int) for df in bond_fea]
    edges_dst = [df.EndAtomIdx.values.astype(int) for df in bond_fea]
    
    edges_src = [edges_src[i] + atom_cumsum[i] for i in range(len(atom_cumsum))]
    edges_dst= [edges_dst[i] + atom_cumsum[i] for i in range(len(atom_cumsum))]

    # to bi-directed
    edges_src = np.concatenate(edges_src)
    edges_dst = np.concatenate(edges_dst)

    all_edges_src = np.concatenate([edges_src, edges_dst])
    all_edges_dst = np.concatenate([edges_dst, edges_src])

    all_atom_fea = np.array(pd.concat(atom_fea))
    all_bond_fea = pd.concat(bond_fea)
    all_bond_fea = all_bond_fea.drop(['BeginAtomIdx', 'EndAtomIdx'], axis=1)
    all_bond_fea = np.array(pd.concat([all_bond_fea, all_bond_fea]))
  
    
    graph = dgl.graph((all_edges_src, all_edges_dst), num_nodes=all_atom_fea.shape[0])
    graph.ndata['feature'] = torch.from_numpy(all_atom_fea).float()
    graph.edata['feature'] = torch.from_numpy(all_bond_fea).float()
    graph = dgl.add_self_loop(graph)
    
    return graph



def read_data(dataset, mode, feature_pkl):
    path = '../data/' + dataset + '/' + mode + '.csv'
    print('preprocessing %s data from %s' % (mode, path))

    # saving all possible values of each attribute (only for training data)
    
    graphs = []

    num_file = sum([1 for i in open(path, 'rb')])
    fail_num = 0

    

    with open(path) as f:
        for _, line in tqdm(enumerate(f), total=num_file):
            if line.split(',')[0] == 'rxn_smiles':
                continue
            rxn, label = line.strip().split(',')
            reactant_smiles, product_smiles = rxn.split('>>')[0], rxn.split('>>')[1]
            reactants = reactant_smiles.split('.')
            products = product_smiles.split('.')
            try:
                reactants_fea = [feature_pkl[i] for i in reactants]
                products_fea = [feature_pkl[i] for i in products]
            except:
                fail_num += 1
                continue

            reactants_mol = [Chem.MolFromSmiles(i) for i in reactants]
            products_mol = [Chem.MolFromSmiles(i) for i in products]

            ###!!!! add H 
            reactants_mol = [Chem.AddHs(i) for i in reactants_mol]
            products_mol = [Chem.AddHs(i) for i in products_mol]

            graphs.append([reactants_mol, reactants_fea, products_mol, products_fea, float(label)])

    print(f'total num of {mode} dataset: ', len(graphs))
    print(f'failed load feature num: {fail_num}')
    return graphs



def load_data(args):
    # if datasets are already cached, skip preprocessing
    print('preprocessing %s dataset' % args.dataset)

    pkl_path = '/home/lichangyu/Mercury/USPTO/data/ECFP_split/ECFP_split_feature.pkl'

    with open(pkl_path, 'rb') as f:
        feature_pkl = pickle.load(f)
    f.close()
    
    train_graphs = read_data(args.dataset, 'train', feature_pkl)
    valid_graphs = read_data(args.dataset, 'valid', feature_pkl)
    test_graphs = read_data(args.dataset, 'test', feature_pkl)

    train_dataset = SmilesDataset(args, 'train', train_graphs)
    valid_dataset = SmilesDataset(args, 'valid', valid_graphs)
    test_dataset = SmilesDataset(args, 'test', test_graphs)
    

    return train_dataset, valid_dataset, test_dataset
