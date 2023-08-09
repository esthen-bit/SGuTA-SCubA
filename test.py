import os
import time
import torch
import numpy as np
from tqdm import tqdm
import config
import myutils

from torch.utils.data import DataLoader

##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"]='2'
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
    if dataset_name == "Xiph":
        from dataset.Xiph import Xiph
        test_set = Xiph(data_root, mode=mode, n_inputs=n_inputs)
        test_loader = DataLoader(test_set, test_batch_size, shuffle=False,  num_workers=args.num_workers, pin_memory=True) 
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

def select_model(model_name,stage):
    if model_name == 'SCubA':
        from model.SCubA import SCubA
        model = SCubA(in_channels=3, out_channels=3, n_feat=64, patch_size=(1,4,4), cube_size=(2,4,4), stage=stage).to(device)
    if model_name == 'SGuTA':
        from model.SGuTA import SGuTA
        model = SGuTA(in_channels=3, out_channels=3, n_feat=64, patch_size=(1,4,4), stage=stage, num_frm=8).to(device)
    if model_name == 'FLAVR':
        from model.FLAVR import FLAVR
        model = FLAVR(n_inputs=4, n_outputs=1, in_channels=3, block='unet_18').to(device)
        
    if model_name == 'VFIT_B':
        from model.VFIT.VFIT_B import VFIT_B
        model = VFIT_B(in_channels=3, n_inputs=4, joinType=args.joinType).to(device)

    if model_name == 'VFIT_S':
        from model.VFIT.VFIT_S import VFIT_S
        model = VFIT_S(in_channels=3, n_inputs=4, joinType=args.joinType).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'# of parameters: {params / 1e6:.2f}M' )
    return model 
def load_model(model ,model_path):
    assert model_path is not None

    loadStateDict = torch.load(model_path, map_location='cuda:0')['state_dict']
    loadStateDict = {k.replace("module.","") : v for k,v in loadStateDict.items()}

    modelStateDict = model.state_dict()

    for k,v in loadStateDict.items():
        if v.shape == modelStateDict[k].shape:
            modelStateDict[k] = v
        else:
            print("Not loading" , k)
    print("Model Loaded Successfully")
    model.load_state_dict(modelStateDict, strict=True)
    print('Current epoch of loaded model:',torch.load(model_path, map_location='cuda:0')["epoch"] )
    
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
    model_name = "SGuTA"#'SCubA'
    stage = 2
    n_inputs = 6
    test_batch_size = 1
    dataset_name = "Xiph"
    data_root= "/mnt/sdb5/xiph/"
    model_path = 'checkpoints/VimeoSepTuplet/SGuTA/SGuTA_stage_2.pth'    
    mode = '2K'
    test_loader = set_dataset(dataset_name=dataset_name, data_root= data_root, n_inputs=n_inputs,test_batch_size=test_batch_size, mode = mode)
    model = select_model(model_name,stage)
    load_model(model, model_path)
    print('Testing %s Model on %s Dataset [%s]!'%(model_name, dataset_name, mode))
    test(args, model, test_loader)


if __name__ == "__main__":
    main(args)