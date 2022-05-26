import os
import argparse
import data_processing_cache
import train
import torch
import warnings

warnings.filterwarnings("ignore")


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='the index of gpu device')

    parser.add_argument('--dataset', type=str, default='ECFP_split', help='dataset name')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--gnn', type=str, default='tag', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=4, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=1024, help='dimension of molecule embeddings')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model', type=bool, default=False, help='save the trained model to disk')
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard/', help='dir of tensorboard log')
    

    args = parser.parse_args()
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = data_processing_cache.load_data(args)
    train.train(args, data, device)
    

if __name__ == '__main__':
    main()
