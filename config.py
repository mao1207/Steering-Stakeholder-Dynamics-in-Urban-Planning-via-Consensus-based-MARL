# config.py
import argparse
import torch
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment and Training Settings.")
    
    # 针对实验的设置
    parser.add_argument('--experiment_mode', type=str, default="random", choices=['actor-critic', 'greedy', 'random'], help="Choose the method: actor-critic or greedy.")
    parser.add_argument('--if_only_top_down', type=bool, default=False)

    # 以下为GNN-DRL的训练设置
    parser.add_argument('--model_mode', type=str, default='GCN')
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--n_hid', type=int, default=512)
    parser.add_argument('--n_inp', type=int, default=19)
    parser.add_argument('--clip', type=int, default=1.0)
    parser.add_argument('--max_lr', type=float, default=5e-4)

    return parser.parse_args()

args = parse_arguments()

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 其他全局变量
max_distance = 1250

script_dir = os.path.dirname(os.path.abspath(__file__))

scale_num = 0.99

locate_one_hot = torch.rand((3, 5)).to(device)