import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
from tqdm import tqdm

import config
import myutils

from torch.utils.data import DataLoader

##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')
print("device", device)

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "ucf101":
    from dataset.ucf101_test import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "gopro":
    from dataset.GoPro import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True, interFrames=args.n_outputs)    
else:
    raise NotImplementedError

# from model.FLAVR_arch_lstm import UNet_3D_3D
# from model.FLAVR_arch import UNet_3D_3D
# from model.FLAVR_arch_lstm_middle import UNet_3D_3D_lstm_middle
print("Building model: %s"%args.model.lower())
# model = UNet_3D_3D(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType)
# from model.UNETR import UNETR
# model = UNETR()
# model = UNet_3D_3D_lstm_middle(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType, upmode=args.upmode)
from model.FlowNetS_unet_noskipflow import FlowNetS_Interpolation
model = FlowNetS_Interpolation()
model = torch.nn.DataParallel(model).to(device)
print("#params" , sum([p.numel() for p in model.parameters()]))

# i added
def make_image(img):
    # img = F.interpolate(img.unsqueeze(0) , (720,1280) , mode="bilinear").squeeze(0)
    q_im = img.data.mul(255.).clamp(0,255).round()
    im = q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return im

def test(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    psnr_list = []
    
    with torch.no_grad():

        for i, (images, gt_image ) in enumerate(tqdm(test_loader)): 

            images = [img_.cuda() for img_ in images]
            gt = [g_.cuda() for g_ in gt_image]

            torch.cuda.synchronize()
            start_time = time.time()
            out = model(images)

            out = torch.cat(out)
            gt = torch.cat(gt)

            torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)

            myutils.eval_metrics(out, gt, psnrs, ssims)

            # i added
            output_image = make_image(out.squeeze(0))
            gt_image = make_image(gt.squeeze(0))
            import imageio
            os.makedirs(f"{args.exp_name}/image/{i}")
            imageio.imwrite(f"{args.exp_name}/frame{i}.png", output_image)
            imageio.imwrite(f"{args.exp_name}/gtframe{i}.png", gt_image) 

            if i > 250:
                break

    print("PSNR: %f, SSIM: %fn" %
          (psnrs.avg, ssims.avg))
    print("Time , " , sum(time_taken)/len(time_taken))

    return psnrs.avg


""" Entry Point """
def main(args):
    
    assert args.load_from is not None

    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
