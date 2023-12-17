import os
import sys
import time
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
import myutils
from loss import Loss
from torch.utils.data import DataLoader

def load_checkpoint(args, model, optimizer , path):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr" , args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)   
elif args.dataset == "gopro":
    from dataset.GoPro import get_loader
    train_loader = get_loader(args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers, test_mode=False, interFrames=args.n_outputs, n_inputs=args.nbr_frame)
    test_loader = get_loader(args.data_root, args.batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True, interFrames=args.n_outputs, n_inputs=args.nbr_frame)
else:
    raise NotImplementedError


# from model.FLAVR_arch import UNet_3D_3D
# from model.
from model.FLAVR_arch_w_inception import UNetWithInception  # Import UNetWithInception
# from model.FLAVR_arch_v2_w_inception import UNetWithInception  # Import UNetWithInception
# from model.FLAVR_arch_w_inception_net import UNet_3D_3D
# from model.FLAVR_arch_w_inception_conv_one import UNetWithInception  # Import UNetWithInception
# from model.FLAVR_arch_v2 import UNet_3D_3D
# from model.UNETR import UNETR

print("Building model: %s"%args.model.lower())
# model = UNETR()
model = UNetWithInception(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType, upmode=args.upmode)
print(model)

from torchsummary import summary
summary(model, input_size = (3, 4, 256, 256))  
# inputs = [torch.randn((2, 3,256,256)).cuda() for _ in range(4)]
# x = model(inputs) 
# print(x[0].shape) 
# num_params = sum(p.numel() for p in model.parameters()) 
# print("Number of parameters in the model:", num_params)
""" Entry Point """
def main(args):

    if args.pretrained:
        ## For low data, it is better to load from a supervised pretrained model
        loadStateDict = torch.load(args.pretrained)['state_dict']
        modelStateDict = model.state_dict()

        for k,v in loadStateDict.items():
            if v.shape == modelStateDict[k].shape:
                print("Loading " , k)
                modelStateDict[k] = v
            else:
                print("Not loading" , k)

        model.load_state_dict(modelStateDict)


if __name__ == "__main__":
    main(args)
