import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of GAN models.')
    parser.add_argument('--model', type=str, default='vgg16', choices = ['vgg16', 'dpp_vgg16'])
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--dataroot', type=str, default='/home/nkim/data', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--save_load', type=bool, default=False)
    parser.add_argument('--save_location', type=str, default=False)
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                     help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                      metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--k_number', type=str, default=[])
    parser.add_argument('--total_models', type=str, default=[])
    parser.add_argument('--best_model_num', type=int, default=1)

    return check_args(parser.parse_args())

def check_args(args):
    try:
        assert args.epochs >= 1
    except:
        print("Number of epohcs must be larger than or equal to one")
    #batch size check
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larget than or equal to one')
    
    try:
        assert len(args.save_dir) == 0
    except:
        print("Enter the save file name")


    return args 
