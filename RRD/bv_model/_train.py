import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold


from RRD.bv_model.data import BVDataset
from RRD.bv_model.featurizer import GenFeatures
from RRD.bv_model.model import BVNet, SaveBestModel

dataset = BVDataset(file_name='./bv_preprocessed_data.json', pre_transform = GenFeatures()) # 

batch_size = 16
epochs = 800

# train, valid, splitting
kf = KFold(n_splits=5, shuffle=True, random_state = 123)

allfold = []
fd = 1
for train_idx, valid_idx in kf.split(range(len(dataset))):
    #train_idx, valid_idx
    
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[valid_idx], batch_size=batch_size)

    print(len(train_loader.dataset), len(val_loader.dataset))
    
    n = 2 #multi-class
    out_channels = n
    in_channels = dataset.data.x.shape[1]
    edge_dim = dataset.data.edge_attr.shape[1]

    pub_args = {'in_channels':in_channels, 'hidden_channels':64, 'out_channels':out_channels,
                'edge_dim':edge_dim, 'num_layers':10, 'dropout_p':0.2, 'batch_norms':None}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BVNet(**pub_args, dropout = 0.1, heads=3).to(device)

    # initialize SaveBestModel class
    saver = SaveBestModel(data_transformer = dataset.smiles_to_data, 
                          save_dir = './trained_model', save_name = 'model_%s.pth' % fd)


    def myloss(out, y):
        l1, l2 = torch.mean((y - out).abs(), 0)
        ls = l1*0.8 + l2*0.2#torch.mean(m1)
        return ls

    optimizer = torch.optim.Adam(model.parameters(), lr=10**-3.5,
                                 weight_decay=10**-5)

    def train(train_loader):
        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x.float(), data.edge_index, data.edge_attr, data.batch)
            loss = myloss(out, data.y[:,:n]) #F.mse_loss(out, data.y[:,:n])

            loss.backward()
            optimizer.step()
            total_loss += float(loss**2) * data.num_graphs
            total_examples += data.num_graphs

            #print(loss)
        return sqrt(total_loss / total_examples)

    @torch.no_grad()
    def test(loader):
        mse = []
        for data in loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index, data.edge_attr, data.batch)
            mse.append(F.mse_loss(out, data.y[:,:n], reduction='none').cpu())
        return float(torch.cat(mse, dim=0).mean().sqrt())

    history = []
    for epoch in range(1, epochs):
        train_rmse = train(train_loader)
        val_rmse = test(val_loader)
        print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f} Val: {val_rmse:.4f} ')
        history.append({'Epoch':epoch, 'train_rmse':train_rmse, 'val_rmse':val_rmse})
        valid_epoch_loss = val_rmse
        saver(valid_epoch_loss, epoch, model, optimizer)

    #save_model
    saver.save()
    fd += 1
    

    alls = []
    for data in val_loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.edge_attr, data.batch)

        y_true = data.y[:, :n].cpu().numpy()
        y_pred = out.cpu().detach().numpy()

        dfres = pd.DataFrame([pd.DataFrame(y_true)[0], pd.DataFrame(y_pred)[0]]).T
        dfres.columns = ['True_BV', 'Pred_BV']

        dfres['batch'] = data.batch.cpu().numpy().reshape(-1,)
        dfres['smiles'] = dfres.batch.map(dict(zip(range(len(data.smiles)),data.smiles)))

        alls.append(dfres)

    dfp = pd.concat(alls)
    allfold.append(dfp)
    
import seaborn as sns
sns.set(style = 'white', font_scale=2)
fig, ax = plt.subplots(figsize=(10,9))
pd.DataFrame(history).set_index('Epoch').plot(ax=ax)
ax.set_ylabel('RMSE')
ax.set_xlabel('Epoch')
fig.savefig('./images/training_history.png',dpi = 300)



from joblib import load, dump
dump(allfold, './train_tmp/allfold.pkl')