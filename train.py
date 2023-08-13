import os
import time
import shutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
from tqdm import tqdm
import importlib
import config
import myutils
from loss import Loss
from torch.optim import Adam

##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()

##### Distributed DataParallel #####
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.distributed.init_process_group('nccl', world_size=args.num_gpu, rank=local_rank)
torch.cuda.set_device(local_rank)
print('Device %s has been set successfully' %torch.distributed.get_rank())

if torch.distributed.get_rank() == args.main_rank:
    dic = vars(args)
    print('####### Setting Parameters are as follows ########')
    for k,v in dic.items():
        print(k,':',v)
    print("#######          That's it                ########")
    
##### log configration #####
save_loc = os.path.join(args.checkpoint_dir , "checkpoints" , args.dataset , args.model, str(args.stage))
if not os.path.exists(save_loc):
    os.makedirs(save_loc)
opts_file = os.path.join(save_loc , "opts.txt")
with open(opts_file , "w") as fh:
    fh.write(str(args))

logger = myutils.initialize_logger(os.path.join(save_loc, 'result.log'), args.model)

##### TensorBoard & Misc Setup #####
writer_loc = os.path.join(args.checkpoint_dir, 'tensorboard_logs', args.dataset, args.model)
writer = SummaryWriter(writer_loc)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
myutils.set_seed(seed=args.random_seed , cuda=args.cuda)

##### Load Dataset #####
train_set = getattr(importlib.import_module("dataset.{}".format(args.dataset)), args.dataset)(args.data_root, is_training=True, n_inputs=args.n_inputs)
train_sampler = DistributedSampler(train_set)
train_loader = DataLoader(train_set, args.batch_size, shuffle=False, sampler=train_sampler,num_workers=args.num_workers, pin_memory=True)

test_set = getattr(importlib.import_module("dataset.{}".format(args.dataset)), args.dataset)(args.data_root, is_training=False, n_inputs=args.n_inputs)    
test_sampler = DistributedSampler(test_set)
test_loader = DataLoader(test_set, args.test_batch_size, shuffle=False, sampler=test_sampler, num_workers=args.num_workers, pin_memory=True) 

# debug_sampler = SubsetRandomSampler(range(1, 10))  
# test_loader = DataLoader(test_set, args.test_batch_size, shuffle=False, sampler=debug_sampler, num_workers=args.num_workers, pin_memory=True) 

if torch.distributed.get_rank() == args.main_rank:
    print("Building model: %s"%args.model)

model = getattr(importlib.import_module("model.{}".format(args.model)), args.model)\
        (in_channels=args.num_channels, n_inputs=args.n_inputs, n_outputs=args.n_outputs, n_feat=args.emb_dim, patch_size=args.patch_size, cube_size=args.cube_size, stage=args.stage, num_scale=args.num_scale).cuda()
        
optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch, eta_min=1e-6)
myutils.load_pretrained_model(args.pretrained, model, args.main_rank, local_rank)
myutils.load_checkpoint(args, model, optimizer ,scheduler, args.checkpoint_path, args.main_rank, local_rank)

if torch.distributed.get_rank() == args.main_rank:
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {params / 1e6:.2f}M' )

model = nn.parallel.DistributedDataParallel(model.cuda(local_rank), device_ids=[local_rank])

##### Define Loss & Optimizer #####
criterion = Loss(args)
## ToDo: Different learning rate schemes for different parameters
def train(args, epoch):
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.train()
    criterion.train()

    t = time.time()
    train_sampler.set_epoch(epoch)
    for i, (images, gt_image) in enumerate(train_loader):

        # Build input batch
        images = [img_.cuda() for img_ in images]
        gt = [gt_.cuda() for gt_ in gt_image]

        # Forward
        optimizer.zero_grad()
        out = model(images)
        
        out = torch.cat(out)
        gt = torch.cat(gt)

        loss, loss_specific = criterion(out, gt)
        
        # Save loss values
        for k, v in losses.items():
            if k != 'total':
                v.update(loss_specific[k].item())
        losses['total'].update(loss.item())

        loss.backward()
        optimizer.step()

        # Calc metrics & print logs
        if i % args.log_iter == 0: 
            myutils.eval_metrics(out, gt, psnrs, ssims)

            if torch.distributed.get_rank() == args.main_rank:
                # print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}\tTime({:.2f})'.format(epoch, i, len(train_loader), losses['total'].avg, psnrs.avg, time.time() - t, flush=True))
                print("Train Epoch: %03d [%05d/%05d] | Loss:%.6f | PSNR: %.3f | SSIM: %.5f | TimeInterval: %03ds | TimeStamp: %s" % (epoch, i, len(train_loader), losses['total'].avg, psnrs.avg, ssims.avg, time.time() - t, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            # Log to TensorBoard
            timestep = epoch * len(train_loader) + i
            writer.add_scalar('Loss/train', loss.data.item(), timestep)
            writer.add_scalar('PSNR/train', psnrs.avg, timestep)
            writer.add_scalar('SSIM/train', ssims.avg, timestep)
            writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], timestep)

            # Reset metrics
            losses, psnrs, ssims = myutils.init_meters(args.loss)
            t = time.time()
def test(args, epoch):
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
        
    t = time.time()
    with torch.no_grad():
        for i, (images, gt_image) in enumerate(tqdm(test_loader)):

            images = [img_.cuda() for img_ in images]
            gt = [gt_.cuda() for gt_ in gt_image]

            out = model(images) ## images is a list of neighboring frames
            out = torch.cat(out)
            gt = torch.cat(gt)

            # Save loss values
            loss, loss_specific = criterion(out, gt)
            
            torch.distributed.barrier()

            loss = myutils.reduce_tensor(loss, args.num_gpu)
            loss_specific['L1'] = myutils.reduce_tensor(loss_specific['L1'], args.num_gpu)

            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            myutils.eval_metrics(out, gt, psnrs, ssims, args.num_gpu)
                    
    # Print progress
    if torch.distributed.get_rank() == args.main_rank:      
        # print("Epoch[%03d/%03d], Loss:%.6f, PSNR: %.3f, SSIM: %.5f, TimeStamp: %s" % (epoch, args.max_epoch, losses['total'].avg, psnrs.avg, ssims.avg, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        logger.info("| Epoch[%03d/%03d] | Loss:%.5f | PSNR: %.5f | SSIM: %.5f |" % (epoch, args.max_epoch, losses['total'].avg, psnrs.avg, ssims.avg))

        # Log to TensorBoard
        timestep = epoch +1
        writer.add_scalar('Loss/test', loss.data.item(), timestep)
        writer.add_scalar('PSNR/test', psnrs.avg, timestep)
        writer.add_scalar('SSIM/test', ssims.avg, timestep)

    return losses['total'].avg, psnrs.avg, ssims.avg


""" Entry Point """
def main(args):
    best_psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        train(args, epoch)
        print("Saving checkpoint for current epoch!")
        myutils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'lr' : optimizer.param_groups[-1]['lr']
        }, save_loc, save_rank=args.main_rank)
        
        # if epoch % 5 == 0:
        _ , psnr, _ = test(args, epoch)

        # save checkpoint
        if (torch.distributed.get_rank() == args.main_rank and psnr > best_psnr):
            best_psnr = max(psnr, best_psnr)
            shutil.copyfile(os.path.join(save_loc , 'checkpoint.pth'), os.path.join(save_loc , 'model_best.pth'))
            print("Best model has been updated!")
        # update optimizer policy
        scheduler.step()

if __name__ == "__main__":
    main(args)
