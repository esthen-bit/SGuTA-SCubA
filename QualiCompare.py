import os
import torch
import cv2
from model.SCubA import SCubA
from model.SGuTA import SGuTA
from torchvision.utils import save_image, make_grid
import torchvision
import numpy as np
import tqdm
import math
from torchvision.io import read_video , write_video
from dataset.transforms import ToTensorVideo , Resize, CenterCropVideo
import torch.nn.functional as F
import argparse
from model.SCubA import SCubA
from model.SGuTA import SGuTA
from model.FLAVR import FLAVR
from model.VFIT.VFIT_B import VFIT_B
from model.VFIT.VFIT_S import VFIT_S
from torchvision import transforms

parser = argparse.ArgumentParser()


# parser.add_argument("--input_dir" , type=str , required=False, default='/mnt/sdb5/SNU-FILM/GOPRO_test/', help="Path/WebURL to input video")
parser.add_argument("--input_dir" , type=str , required=False, default='/mnt/sdb5/Davis_test/', help="Path/WebURL to input video")
# parser.add_argument("--input_dir" , type=str , required=False, default='/mnt/sdb5/xiph/', help="Path/WebURL to input video")
parser.add_argument("--output_sub_path" , type=str , required=False, default='QualiCompare_1234', help="Path/WebURL to input video")
parser.add_argument('--nbr_frame' , type=int , default=6)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--patch_size', type=tuple, default=(1,4,4))
parser.add_argument('--stage', type=int, default=3)
parser.add_argument('--joinType' , choices=["concat" , "add" , "none"], default="concat")
parser.add_argument('--num_channels' , type=int , default=3)
parser.add_argument('--model_path',  type=str, default="saved_models_final/vimeo90K_septuplet/ScubA_Hola/model_best.pth")
parser.add_argument("--factor" , type=int , required=False , default=2, choices=[2,4,8,16,32] , help="How much interpolation needed. 2x/4x/8x.")
parser.add_argument("--codec" , type=str , help="video codec" , default="mpeg4")
parser.add_argument("--output_ext" , type=str , help="Output video format" , default=".avi")
parser.add_argument("--input_ext" , type=str, help="Input video format", default=".mp4")
parser.add_argument("--t_downscale" , type=float , help="temporal Downscale" , default=1)
parser.add_argument("--downscale" , type=float , help="spatial Downscale" , default=1)
parser.add_argument("--output_fps" , type=int , help="Target FPS" , default=30)
parser.add_argument("--is_folder" ,default='/home/esthen/Datasets/Davis_test/bear/', action="store_true" )
parser.add_argument("--device" , type=str ,default="cuda:0")
args = parser.parse_args()


def combine_image(out_raw, out_sguta,out_scuba,out_flavr, out_vfits,out_vfitb):
    image_tensors = [out_raw, out_sguta,out_scuba,out_vfitb, out_vfits, out_flavr]
    grid_image = make_grid(image_tensors, nrow=3, padding=10)
    return grid_image
def loadModel(model, checkpoint):
    
    saved_state_dict = torch.load(checkpoint)['state_dict']
    saved_state_dict = {k.partition("module.")[-1]:v for k,v in saved_state_dict.items()}
    model.load_state_dict(saved_state_dict)

def write_video_cv2(frames , video_name , fps , sizes):

    out = cv2.VideoWriter(video_name,cv2.CAP_OPENCV_MJPEG,cv2.VideoWriter_fourcc('M','J','P','G'), fps, sizes)

    for frame in frames:
        out.write(frame)

def make_image(img):
    q_im = img.data.mul(255.).clamp(0,255).round()
    im = q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

def files_to_videoTensor(path , time_downscale=1):
    from PIL import Image
    files = sorted(os.listdir(path))[::time_downscale]#[:80]
    print(len(files))
    T = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((720,844))
                # transforms.Resize((768, 768), interpolation=Image.BILINEAR)
            ])
    # images = [torch.Tensor(np.asarray(Image.open(os.path.join(input_video , f)).convert("RGB"))).type(torch.uint8) for f in files]
    videoTensor = [T(Image.open(os.path.join(input_video , f))) for f in files]
    
    # print("Shape of Input Video Tensor is:",images[0].shape)
    # videoTensor = torch.stack(images)
    return videoTensor

def video_to_tensor(video):

    videoTensor , _ , md = read_video(video)
    fps = md["video_fps"]
    print(fps)
    return videoTensor

def video_transform(videoTensor , downscale=1):
    
    T , H , W = videoTensor.size(0), videoTensor.size(1) , videoTensor.size(2)
    downscale = int(downscale * 64)
    resizes = 64*(H//downscale) , 64*(W//downscale)
    transforms = torchvision.transforms.Compose([ToTensorVideo() , CenterCropVideo(resizes)])
    videoTensor = transforms(videoTensor)
    print("Resizing to %dx%d"%(resizes[0] , resizes[1]) )
    return videoTensor , resizes

os.environ["CUDA_VISIBLE_DEVICES"]='1'
input_ext = args.input_ext

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


theme = sorted(os.listdir(args.input_dir))
for i in range(len(theme)):
    print("Process video [{}]".format(theme[i]))
    input_video = os.path.join(args.input_dir,theme[i])
    if input_video.endswith("/"):
        video_name = input_video.split("/")[-2].split(input_ext)[0]
    else:
        video_name = input_video.split("/")[-1].split(input_ext)[0]

    frames = files_to_videoTensor(input_video , args.t_downscale)
    
    # videoTensor , resizes = video_transform(videoTensor , args.downscale)
    # frames = torch.unbind(videoTensor , 1)
    # n_inputs = len(frames)
    with torch.no_grad():    
        for i in range(int(math.log(args.factor, 2))):
            torch.cuda.empty_cache()
            idxs = torch.Tensor(range(len(frames))).type(torch.long).view(1,-1).unfold(1,size=args.nbr_frame,step=1).squeeze(0)
            outputs = [] ## store the input and interpolated frames
            outputs.append(combine_image(frames[0],frames[0],frames[0],frames[0],frames[0],frames[0]))
            outputs.append(combine_image(frames[1],frames[1],frames[1],frames[1],frames[1],frames[1]))
            outputs.append(combine_image(frames[2],frames[2],frames[2],frames[2],frames[2],frames[2]))
            for i in tqdm.tqdm(range(len(idxs))):
                torch.cuda.empty_cache()
                idxSet = idxs[i]
                inputs_1 = [frames[idx_].to(args.device).unsqueeze(0) for idx_ in idxSet]
                # inputs_2 = [inputs_1[0],inputs_1[2],inputs_1[3],inputs_1[5]]
                inputs_2 = [inputs_1[1],inputs_1[2],inputs_1[3],inputs_1[4]]

                out_vfits = vfit_s(inputs_2)[0].squeeze(0)
                out_vfitb = vfit_b(inputs_2)[0].squeeze(0)    
                out_sguta = sguta(inputs_1)[0].squeeze(0)  
                out_scuba = scuba(inputs_1)[0].squeeze(0)  
                out_flavr = flavr(inputs_2)[0].squeeze(0) 
                out_raw = inputs_1[2].squeeze(0) 
                combined_image = combine_image(out_raw, out_sguta, out_scuba,out_flavr, out_vfits, out_vfitb)
                outputs.append(combined_image.cpu())
                
                outputs.append(combine_image(inputs_1[3].squeeze(0).cpu(),inputs_1[3].squeeze(0).cpu(),inputs_1[3].squeeze(0).cpu(),inputs_1[3].squeeze(0).cpu(),inputs_1[3].squeeze(0).cpu(),inputs_1[3].squeeze(0).cpu()))                        
            outputs.append(combine_image(frames[-3],frames[-3],frames[-3],frames[-3],frames[-3],frames[-3]))
            outputs.append(combine_image(frames[-2],frames[-2],frames[-2],frames[-2],frames[-2],frames[-2]))
            outputs.append(combine_image(frames[-1],frames[-1],frames[-1],frames[-1],frames[-1],frames[-1]))
            frames = outputs
    output_dir = args.input_dir.replace('sdb5/','sdb5/'+f"{args.output_sub_path}/")
    image_save_path = os.path.join(output_dir, 'Sequence' ,video_name)
    
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
        
    for i, im_ in enumerate(outputs):
        save_image(im_, os.path.join(image_save_path, "{:05d}.png".format(i)))
    new_video = [make_image(im_) for im_ in outputs]
    path_output_video = os.path.join(output_dir, "Video")
    if not os.path.exists(path_output_video):
        os.makedirs(path_output_video)
    write_video_cv2(new_video , os.path.join(path_output_video, video_name + f"_{args.factor}x" + str(args.output_ext)) , args.output_fps , (outputs[0].shape[-1] , outputs[0].shape[-2]))
    print("Video [", path_output_video.split(".")[0] + ".avi", "] has been successfully written")

# if __name__ == '__main__':

    