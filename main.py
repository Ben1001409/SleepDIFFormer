import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from timeit import default_timer as timer
from utils import *
import random
from datasets.dataset import LoadDataset
from trainer import Trainer
from utils.visualize import Visualization
from sklearn.model_selection import ParameterGrid
datasets = [
    'sleep-edfx',
    'HMC',
    'ISRUC',
    'SHHS1',
    'P2018',
]


def main():
    seed = 0
    cuda_id = 4
    setup_seed(seed)
    torch.cuda.set_device(cuda_id)
    accs, f1s = [], []
    for dataset_name in datasets:
        parser = argparse.ArgumentParser(description='SleepDiFFormer')
        parser.add_argument('--target_domains', type=str, default=dataset_name, help='target_domains')
        parser.add_argument('--project_name',type=str,default="SleepDIFFormer",help='Project name in wandb')
        parser.add_argument('--run_name',type=str,default="experiment1",help='experiment name in wandb')
        # parser.add_argument('--seed', type=int, default=443, help='random seed (default: 0)')
        # parser.add_argument('--cuda', type=int, default=4, help='cuda number (default: 1)')
        parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
        parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
        parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='dropout')
        parser.add_argument('--datasets_dir', type=str, default='/home/tester1/YT/Ben/SleepDiFFormer/data/datasets/GSS_datasets', help='datasets_dir')
        #parser.add_argument('--model_dir', type=str, default='/home/tester1/YT/Ben/SleepDiFFormer/results3/diff_attn', help='model_dir')
        parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
        parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
        #parser.add_argument('--model_path',type=str, default='/home/tester1/YT/Ben/SleepDiFFormer/result110_test_linearinexog_learnablepe_8headinseqencoderglobal_8headintimexer_patch_len50_bz16_concat_dmodel128_corrected_4layer_withdecreasingkernelsize_modifiedshhs1_4layer_4head_4975/sleep-edfx_tacc_0.78186_tf1_0.72443.pth', help='model_path')
        parser.add_argument('--log_file',type=str, default='SleepDIFFormer', help='log file name')
        parser.add_argument('--return_attention',type=bool, default=False, help='whether to return attention')
        parser.add_argument('--plot_attention',type=bool, default=False, help='whether to plot attention')
        parser.add_argument('--optimizer',type=str, default="Adam", help='whether to use RAdam or Adam')
        parser.add_argument('--patience',type=int, default=5, help='early stopping criteria')
        parser.add_argument('--min_delta',type=float, default=0.0,help='minimum change in monitored value to qualify as improvement')
        parser.add_argument('--accumulation_steps',type=int, default=1,help='accumulation steps in gradient accumuulation')
        parser.add_argument('--num_heads',type=int, default=4,help='number of heads in transformer')
        parser.add_argument('--num_layers',type=int, default=4,help='number of layers of encoder and decoder layer')
        parser.add_argument('--use_normal',type=bool, default=False, help='whether to use normal attention or diff attention')
        parser.add_argument("--d_model",type=int,default=128,help="dimension of transformer model")
        parser.add_argument("--patch_len",type=int,default=100,help="patch length for each epoch")
        parser.add_argument("--d_ff",type=int,default=512,help="feedforward dimension")
        parser.add_argument("--use_focal",type=bool,default=False,help="whether to use focal loss")
        params = parser.parse_args()
        print(params)

        trainer = Trainer(params)
        test_acc, test_f1 = trainer.train()
        accs.append(test_acc)
        f1s.append(test_f1)
    print(accs)
    print(f1s)
    print(np.mean(accs), np.mean(f1s))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()

