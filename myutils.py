# from https://github.com/myungsub/CAIN/blob/master/utils.py, 
# but removed the errenous normalization and quantization steps from computing the PSNR.

from pytorch_msssim import ssim_matlab as calc_ssim
import math
import os
import torch
import numpy as np
import random
import torch.distributed as dist
from torchvision import transforms
import torch.nn.functional as F
import logging


def set_seed(seed=None, cuda=False): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed) 

def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims

def eval_metrics(output, gt, psnrs, ssims, num_gpu=None):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        if num_gpu is not None:
            dist.barrier()
            psnr = reduce_tensor(psnr, num_gpu)
        psnrs.update(psnr)

        ssim = calc_ssim(output[b].unsqueeze(0).clamp(0,1), gt[b].unsqueeze(0).clamp(0,1) , val_range=1.)
        if num_gpu is not None:
            dist.barrier()
            ssim = reduce_tensor(ssim, num_gpu)
        ssims.update(ssim)

def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_type = l.split('*')
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    psnr =  -10 * torch.log10(diff)
    return psnr


def load_checkpoint(args, model, optimizer, scheduler, checkpoint_path, main_rank, local_rank):
    if checkpoint_path is not None:
        if torch.distributed.get_rank() == main_rank:
            print("loading checkpoint %s" % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(local_rank))
        loadStateDict = checkpoint['state_dict']
        # loadStateDict = {'module.'+k : v for k,v in loadStateDict.items()}
        epoch = checkpoint['epoch']
        args.start_epoch = epoch + 1

        modelStateDict = model.state_dict()
        for k,v in loadStateDict.items():
            if v.shape == modelStateDict[k].shape:
                if torch.distributed.get_rank() == main_rank:
                    print("Loading " , k)
                    modelStateDict[k] = v
            else:
                if torch.distributed.get_rank() == main_rank:
                   print("Not loading" , k)        
        model.load_state_dict(loadStateDict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        lr = checkpoint.get("lr" , args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if torch.distributed.get_rank() == main_rank:
            print('Checkpoint for epoch %s has been loaded' %checkpoint['epoch'])
        return model, epoch, optimizer, scheduler
    else:
        if torch.distributed.get_rank() == main_rank:
            print('No checkpoint need to be loaded')        

def load_pretrained_model(pretrained, model, main_rank, local_rank):
    if pretrained is not None:
        ## For low data, it is better to load from a supervised pretrained model
        loadStateDict = torch.load(pretrained, map_location='cuda:{}'.format(local_rank))['state_dict']
        # loadStateDict = {'module.'+k : v for k,v in loadStateDict.items()}

        modelStateDict = model.state_dict()

        for k,v in loadStateDict.items():
            if v.shape == modelStateDict[k].shape:
                print("Loading " , k)
                modelStateDict[k] = v
            else:
                print("Not loading" , k)

        model.load_state_dict(modelStateDict)
        if torch.distributed.get_rank() == main_rank:
            print('Pretrained model has been loaded from: %s' %pretrained)
        return model

    else:
        if torch.distributed.get_rank() == main_rank:
            print('No pretrained model need to be loaded')

def save_checkpoint(state, directory, save_rank, filename='checkpoint.pth' ):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory , filename)
    if torch.distributed.get_rank() == save_rank:
        torch.save(state, filename)

    
def set_seed(seed=None, cuda=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        
def log_tensorboard(writer, loss, psnr, ssim, lpips, lr, timestep, mode='train'):
    writer.add_scalar('Loss/%s/%s' % mode, loss, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)
        
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def initialize_logger(file_dir,logger_name):
    logger = logging.getLogger(logger_name)
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger