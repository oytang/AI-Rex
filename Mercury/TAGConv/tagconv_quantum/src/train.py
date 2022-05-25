import os
import torch
import torch.nn as nn
import pickle
import numpy as np
from model import MyModel
from copy import deepcopy
from dgl.dataloading import GraphDataLoader
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter


def train(args, data, device):
    train_data, valid_data, test_data = data
    node_dim = train_data[0][0].ndata['feature'].shape[1]
    edge_dim = train_data[0][0].edata['feature'].shape[1]
    model = MyModel(args.gnn, args.layer, node_dim, edge_dim, args.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    writer = SummaryWriter(args.tensorboard_dir)


    criterion = nn.BCELoss()

    if torch.cuda.is_available():
        model = model.to(device)

    best_model_params = None
    best_val_f1 = 0
    print('start training\n')

    print('initial case:')
    model.eval()
    evaluate(model, 'valid', valid_data, criterion, args, device)
    evaluate(model, 'test', test_data, criterion, args, device)
    print()

    for i in range(args.epoch):
        print('epoch %d:' % i)

        # train
        model.train()
        all_label = []
        all_pred = []
        loss_sum = 0
        for reactant_graphs, product_graphs, labels in train_dataloader:
            reactant_graphs = reactant_graphs.to(device)
            product_graphs = product_graphs.to(device)
            labels = labels.to(device).unsqueeze(1)
            preds = model(reactant_graphs, product_graphs)
            
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            all_pred.append(preds.detach().cpu().numpy())
            all_label.append(labels.detach().cpu().numpy())
            loss_sum += loss.item() * labels.shape[0]
        accuracy, precision, recall, f1_score, roc_auc_score = cal_metrics(all_pred, all_label)
        print('train  loss_sum: %.4f  accuracy: %.4f  precision: %.4f  recall: %.4f  f1_score: %.4f  roc_auc_score: %.4f' % (loss_sum, accuracy, precision, recall, f1_score, roc_auc_score))
        writer.add_scalar('train/loss', loss_sum, i)
        writer.add_scalar('train/accuracy', accuracy, i)
        writer.add_scalar('train/precision', precision, i)
        writer.add_scalar('train/recall', recall, i)
        writer.add_scalar('train/f1_score', f1_score, i)
        writer.add_scalar('train/roc_auc_score', roc_auc_score, i)



        # evaluate on the validation set
        val_accuracy, val_precision, val_recall, val_f1, val_roc_auc_score, val_loss_sum = evaluate(model, 'valid', valid_data, criterion, args, device)
        writer.add_scalar('valid/loss', val_loss_sum, i)
        writer.add_scalar('valid/accuracy', val_accuracy, i)
        writer.add_scalar('valid/precision', val_precision, i)
        writer.add_scalar('valid/recall', val_recall, i)
        writer.add_scalar('valid/f1_score', val_f1, i)
        writer.add_scalar('valid/roc_auc_score', val_roc_auc_score, i)


        test_accuracy, test_precision, test_recall, test_f1_score, test_roc_auc_score, test_loss_sum = evaluate(model, 'test', test_data, criterion, args, device)
        writer.add_scalar('test/loss', test_loss_sum, i)
        writer.add_scalar('test/accuracy', test_accuracy, i)
        writer.add_scalar('test/precision', test_precision, i)
        writer.add_scalar('test/recall', test_recall, i)
        writer.add_scalar('test/f1_score', test_f1_score, i)
        writer.add_scalar('test/roc_auc_score', test_roc_auc_score, i)

        # save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_params = deepcopy(model.state_dict())

        print()

    # evaluation on the test set
    print('final results on the test set:')
    model.load_state_dict(best_model_params)
    evaluate(model, 'test', test_data, criterion, args, device)
    print()

    # save the model, hyperparameters, and feature encoder to disk
    if args.save_model:
        if not os.path.exists('./saved/'):
            print('creating directory: ./saved/')
            os.mkdir('./saved/')

        directory = './saved/%s_%s_%d' % (args.dataset, args.gnn, args.dim)
        # directory += '_' + time.strftime('%Y%m%d%H%M%S', time.localtime())
        if not os.path.exists(directory):
            os.mkdir(directory)

        print('saving the model to directory: %s' % directory)
        torch.save(best_model_params, directory + '/model.pt')
        with open(directory + '/hparams.pkl', 'wb') as f:
            hp_dict = {'gnn': args.gnn, 'layer': args.layer, 'node_dim': node_dim, 'edge_dim': edge_dim, 'dim': args.dim}
            pickle.dump(hp_dict, f)
        


def evaluate(model, mode, data, criterion, args, device):
    model.eval()
    with torch.no_grad():
        all_label = []
        all_pred = []
        loss_sum = 0
        valid_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        for reactant_graphs, product_graphs, labels in valid_dataloader:
            reactant_graphs = reactant_graphs.to(device)
            product_graphs = product_graphs.to(device)
            labels = labels.to(device).unsqueeze(1)
            preds = model(reactant_graphs, product_graphs)
            
            loss = criterion(preds, labels)
            all_pred.append(preds.detach().cpu().numpy())
            all_label.append(labels.detach().cpu().numpy())
            loss_sum += loss.item() * labels.shape[0]


        
        accuracy, precision, recall, f1_score, roc_auc_score = cal_metrics(all_pred, all_label)
        print('%s  loss_sum: %.4f  accuracy: %.4f  precision: %.4f  recall: %.4f  f1_score: %.4f  roc_auc_score: %.4f' % (mode, loss_sum, accuracy, precision, recall, f1_score, roc_auc_score))
        return accuracy, precision, recall, f1_score, roc_auc_score, loss_sum


def cal_metrics(preds, labels, threshold=0.5):
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds_int = np.array([int(i>threshold) for i in preds])
    accuracy = metrics.accuracy_score(labels, preds_int)
    precision = metrics.precision_score(labels, preds_int)
    recall = metrics.recall_score(labels, preds_int)
    f1_score = metrics.f1_score(labels, preds_int)
    roc_auc_score = metrics.roc_auc_score(labels, preds)

    return accuracy, precision, recall, f1_score, roc_auc_score
