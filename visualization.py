from dataset.VimeoSepTuplet import VimeoSepTuplet
from dataset.Davis_test import Davis
from dataset.SNU_FILM import SNUFILM
from model.SCubA import SCubA
from model.SGuTA import SGuTA
from model.FLAVR import FLAVR
from model.VFIT.VFIT_B import VFIT_B
from model.VFIT.VFIT_S import VFIT_S
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torch
import os
import myutils


def init_dataset(dataset,mode="hard"):
    if dataset == 'Vimeo_90k':
        test_set = VimeoSepTuplet('/mnt/sdb5/vimeo_septuplet', is_training=False)
    if dataset == 'Davis':
        test_set = Davis("/home/esthen/Datasets/Davis_test/")
    if dataset == 'SNUFILM':
        test_set = SNUFILM('/mnt/sdb5/SNU-FILM',mode=mode, n_inputs=6)
    print("Length of %s test-set %s:"%(dataset,len(test_set)))
    return test_set

def combine_image(out_raw, out_sguta,out_scuba,out_flavr, out_vfits,out_vfitb):
    image_tensors = [out_raw, out_sguta,out_scuba,out_vfitb, out_vfits, out_flavr]
    grid_image = make_grid(image_tensors, nrow=3, padding=10)
    return grid_image

def visual(test_set, device,save_loc,logger):
    scuba = SCubA(in_channels=3, n_outputs=1, n_feat=64, patch_size=(1,4,4), cube_size=(2,4,4), stage=3).to(device)
    scuba.load_state_dict(torch.load('checkpoints/VimeoSepTuplet/SCubA/SCubA_stage_3.pth')["state_dict"] , strict=True)
    scuba.eval()
    sguta = SGuTA(in_channels=3, out_channels=3, n_feat=64, patch_size=(1,4,4), stage=3, num_frm=8).to(device)
    sguta.load_state_dict(torch.load('checkpoints/VimeoSepTuplet/SGuTA/SGuTA_stage_3.pth')["state_dict"] , strict=True)
    sguta.eval()
    flavr = FLAVR(n_inputs=4, n_outputs=1, in_channels=3).to(device)
    loadStateDict = torch.load('checkpoints/VimeoSepTuplet/FLAVR/FLAVR_2x.pth', map_location=device)['state_dict']
    loadStateDict = {k.replace('module.', '') : v for k,v in loadStateDict.items()}
    flavr.load_state_dict(loadStateDict, strict=True)
    flavr.eval()
    vfit_b = VFIT_B(in_channels=3, n_inputs=4).to(device)
    loadStateDict = torch.load('checkpoints/VimeoSepTuplet/VFIT/VFIT_B_best.pth', map_location=device)['state_dict']
    loadStateDict = {k.replace('module.', '') : v for k,v in loadStateDict.items()}
    vfit_b.load_state_dict(loadStateDict , strict=True)
    vfit_b.eval()
    vfit_s = VFIT_S(in_channels=3, n_inputs=4).to(device) 
    loadStateDict = torch.load('checkpoints/VimeoSepTuplet/VFIT/VFIT_S_best.pth', map_location=device)['state_dict']
    loadStateDict = {k.replace('module.', '') : v for k,v in loadStateDict.items()}
    vfit_s.load_state_dict(loadStateDict , strict=True)
    vfit_s.eval() 
    
    
    for idx in tqdm(range(len(test_set))):
        input = [img.unsqueeze(0).to(device) for img in test_set[idx][0]]
        input_1 = [input[0],input[2],input[3],input[5]]
        snufilm_gt = [gt.unsqueeze(0).to(device) for gt in test_set[idx][1]]
        snufilm_overlay = input[1]*0.5 + input[4]*0.5
        with torch.no_grad():
            
            
            gt = torch.cat(snufilm_gt)
            
            out_scuba = scuba(input)
            out_scuba = torch.cat(out_scuba)

            
            out_sguta = sguta(input)
            out_sguta = torch.cat(out_sguta)   

            out_flavr = flavr(input_1)
            out_flavr = torch.cat(out_flavr) 
        
            out_vfit_s = vfit_s(input_1)
            out_vfit_s = torch.cat(out_vfit_s) 
            out_vfit_b = vfit_b(input_1)
            out_vfit_b = torch.cat(out_vfit_b) 

            psnr_scuba = myutils.calc_psnr(out_scuba,gt)
            psnr_sguta = myutils.calc_psnr(out_sguta,gt)
            psnr_vfit_s = myutils.calc_psnr(out_vfit_s,gt)
            psnr_vfit_b = myutils.calc_psnr(out_vfit_b,gt)
            psnr_flavr = myutils.calc_psnr(out_flavr,gt)
            if psnr_scuba > psnr_sguta > psnr_vfit_b+0.5 > psnr_vfit_s+0.5 > psnr_flavr+0.5:
                combined_image = combine_image(gt.squeeze(0), out_sguta.squeeze(0), out_scuba.squeeze(0),out_flavr.squeeze(0), out_vfit_s.squeeze(0), out_vfit_b.squeeze(0))
                save_image(combined_image, os.path.join(save_loc,'{}_Combined.png'.format(idx)))
                save_image(snufilm_overlay.squeeze(0), os.path.join(save_loc,'{}_overlay.png'.format(idx)))
                logger.info("| Idx:%04d | SGuTA:%.3f | SCubA:%.3f | FLAVR: %.3f | VFIT-S: %.3f | VFIT-B: %.3f| " % (idx, psnr_sguta, psnr_scuba, psnr_flavr,psnr_vfit_s,psnr_vfit_b))

                print("images of index [{}] saved".format(idx))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    device = 'cuda:0'
    dataset='SNUFILM'
    mode = 'hard'
    if dataset=='SNUFILM':
        test_set = init_dataset(dataset=dataset,mode=mode)
        save_loc = os.path.join('/mnt/sdb5/Visual_Compare',dataset,mode)
    else:
        test_set = init_dataset(dataset=dataset,mode=mode)
        save_loc = os.path.join('/mnt/sdb5/Visual_Compare',dataset)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    logger = myutils.initialize_logger(os.path.join(save_loc, 'result.log'), 'QuliCompare')
    visual(test_set,device,save_loc,logger)

