import os
import time
import torch
import numpy as np
from tqdm import tqdm
import config
import myutils
from model.SCubA import SCubA
from model.SGuTA import SGuTA
from model.FLAVR import FLAVR
from model.VFIT.VFIT_B import VFIT_B
from model.VFIT.VFIT_S import VFIT_S
from torch.utils.data import DataLoader

##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"]='1'
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)
def set_dataset(dataset_name, data_root,n_inputs=6, test_batch_size=8,mode='easy'):
            
    if dataset_name == "SNU_FILM":
        from dataset.SNU_FILM import SNUFILM
        dataset = SNUFILM(data_root, mode=mode, n_inputs=n_inputs)
        test_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if args.dataset == "PHSPD":
        from dataset.PHSPD import PHSPD
        test_set = PHSPD(args.data_root, is_training=False)
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) 

    if dataset_name == "vimeo90K_septuplet":
        from dataset.VimeoSepTuplet import VimeoSepTuplet   
        test_set = VimeoSepTuplet(data_root, is_training=False, n_inputs=n_inputs)
        test_loader = DataLoader(test_set, test_batch_size, shuffle=False,  num_workers=args.num_workers, pin_memory=True) 
        # debug_sampler = sampler.SubsetRandomSampler(range(1, 10))  
        # test_loader = DataLoader(test_set, test_batch_size, shuffle=False, sampler=debug_sampler, num_workers=args.num_workers, pin_memory=True) 

    if dataset_name == "Davis_test":
        from dataset.Davis_test import Davis
        test_set = Davis(data_root, n_inputs=n_inputs)
        test_loader = DataLoader(test_set, test_batch_size, shuffle=False,  num_workers=args.num_workers, pin_memory=True) 
        
    if dataset_name == "Middlebury_other":
        from dataset.Middlebury_other import Middlebury
        test_set = Middlebury(data_root, n_inputs=n_inputs)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
    if dataset_name == "ucf101":
        from dataset.ucf101_test import get_loader
        data_root = "/home/esthen/Datasets/ucf101_extracted/"
        test_loader = get_loader(data_root, test_batch_size,  num_workers=args.num_workers,n_inputs=n_inputs)
    if dataset_name == "gopro":
        from dataset.GoPro import get_loader
        test_loader = get_loader(data_root, test_batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True, interFrames=args.n_outputs)    
    return test_loader

def define_load_model(model_name):
    scuba = SCubA(in_channels=3, n_outputs=1, n_feat=64, patch_size=(1,4,4), cube_size=(2,4,4), stage=3).to(args.device)
    scuba.load_state_dict(torch.load('checkpoints/VimeoSepTuplet/SCubA/SCubA_stage_3.pth')["state_dict"] , strict=True)
    scuba.eval()
    sguta = SGuTA(in_channels=3, out_channels=3, n_feat=64, patch_size=(1,4,4), stage=3, num_frm=8).to(args.device)
    sguta.load_state_dict(torch.load('checkpoints/VimeoSepTuplet/SGuTA/SGuTA_stage_3.pth')["state_dict"] , strict=True)
    sguta.eval()
    flavr = FLAVR(n_inputs=4, n_outputs=1, in_channels=3).to(args.device)
    loadStateDict = torch.load('checkpoints/VimeoSepTuplet/FLAVR/FLAVR_2x.pth', map_location=args.device)['state_dict']
    loadStateDict = {k.replace('module.', '') : v for k,v in loadStateDict.items()}
    flavr.load_state_dict(loadStateDict, strict=True)
    flavr.eval()
    vfit_b = VFIT_B(in_channels=3, n_inputs=4, joinType=args.joinType).to(args.device)
    loadStateDict = torch.load('checkpoints/VimeoSepTuplet/VFIT/VFIT_B_best.pth', map_location=args.device)['state_dict']
    loadStateDict = {k.replace('module.', '') : v for k,v in loadStateDict.items()}
    vfit_b.load_state_dict(loadStateDict , strict=True)
    vfit_b.eval()
    vfit_s = VFIT_S(in_channels=3, n_inputs=4, joinType=args.joinType).to(args.device) 
    loadStateDict = torch.load('checkpoints/VimeoSepTuplet/VFIT/VFIT_S_best.pth', map_location=args.device)['state_dict']
    loadStateDict = {k.replace('module.', '') : v for k,v in loadStateDict.items()}
    vfit_s.load_state_dict(loadStateDict , strict=True)
    vfit_s.eval() 

def test(args, model, test_loader):
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

    print("PSNR: %.2f, SSIM: %.3f ,Time: %.3f"%
          (psnrs.avg, ssims.avg, sum(time_taken)/len(time_taken)))


    return psnrs.avg


""" Entry Point """
def main(args):
    n_inputs = 4
    test_batch_size = 1
    dataset_name = "SNU_FILM"
    data_root = '/mnt/sdb5/SNU-FILM/'
    mode = 'hard'
    test_loader = set_dataset(dataset_name=dataset_name, data_root= data_root, n_inputs=n_inputs,test_batch_size=test_batch_size, mode = mode)
    define_load_model()
    load_model(model, model_path)
    test(args, model, test_loader)


if __name__ == "__main__":
    main(args)
