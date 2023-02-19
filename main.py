import argparse
from utils.Context import ContextManager, DatasetManager

import torch
import numpy as np
import random 
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
setup_seed(20210823)

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='experiment name', default='default')
parser.add_argument('--description', type=str, help='exp details, used for log name', default='default')
parser.add_argument('--workspace', type=str, default='./workspace')
parser.add_argument('--dataset_name', type=str, default='mind')

parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--tb', type=bool, help='whether use tensorboard', default=False)
parser.add_argument('--train_tb', type=bool, help='whether use tensorboard in training', default=False)
parser.add_argument('--verbose', type=bool, help='whether save model paremeters in tensorborad', default=False)
parser.add_argument('--model', type=str, help='which model to use', default='')

args = parser.parse_args()

print(args)

flags_obj = args
cm = ContextManager(flags_obj)
dm = DatasetManager(flags_obj)

dm.show()
new_config = None
trainer = cm.set_trainer(flags_obj, cm, dm, new_config)

trainer.train()
trainer.test()