import os
import torch
import cv2
from model.SCubA import SCubA
from model.SGuTA import SGuTA
from torchvision.utils import save_image
import torchvision
import numpy as np
import tqdm
import math
from torchvision.io import read_video , write_video
from dataset.transforms import ToTensorVideo , Resize, CenterCropVideo
import torch.nn.functional as F
import argparse
import importlib

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir" , type=str , required=False, default='polar_interpolation/', help="Path/WebURL to input video")
parser.add_argument("--output_dir" , type=str , required=False, default='polar_result/32x', help="Path/WebURL to input video")
parser.add_argument('--model',  type=str, default="SCubA")
parser.add_argument('--nbr_frame' , type=int , default=6)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--patch_size', type=tuple, default=(1,4,4))
parser.add_argument('--stage', type=int, default=3)
parser.add_argument('--joinType' , choices=["concat" , "add" , "none"], default="concat")
parser.add_argument('--num_channels' , type=int , default=3)
parser.add_argument('--model_path',  type=str, default="saved_models_final/vimeo90K_septuplet/ScubA_Hola/model_best.pth")
parser.add_argument("--factor" , type=int , required=False , default=4, choices=[2,4,8,16,32] , help="How much interpolation needed. 2x/4x/8x.")
parser.add_argument("--codec" , type=str , help="video codec" , default="mpeg4")
parser.add_argument("--load_model" , required=False , default='FLAVR_2x.pth',type=str , help="path for stored model")
parser.add_argument("--output_ext" , type=str , help="Output video format" , default=".avi")
parser.add_argument("--input_ext" , type=str, help="Input video format", default=".mp4")
parser.add_argument("--downscale" , type=float , help="Downscale input res. for memory" , default=1)
parser.add_argument("--output_fps" , type=int , help="Target FPS" , default=30)
parser.add_argument("--is_folder" ,default='/home/esthen/Datasets/Davis_test/bear/', action="store_true" )
parser.add_argument("--device" , type=str ,default="cuda:0")
args = parser.parse_args()


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

def files_to_videoTensor(path , downscale=1.):
    from PIL import Image
    files = sorted(os.listdir(path))
    print(len(files))
    images = [torch.Tensor(np.asarray(Image.open(os.path.join(input_video , f)).convert("RGB"))).type(torch.uint8) for f in files]
    
    print("Shape of Input Video Tensor is:",images[0].shape)
    videoTensor = torch.stack(images)
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


input_ext = args.input_ext
model = getattr(importlib.import_module("model.{}".format(args.model)), args.model)(in_channels=args.num_channels, out_channels=args.num_channels,n_feat=args.emb_dim, patch_size=args.patch_size,stage=args.stage).cuda()
model.load_state_dict(torch.load(args.model_path)["state_dict"] , strict=True)
model.eval()
theme = sorted(os.listdir(args.input_dir))
for i in range(len(theme)):
    print("Process video [{}]".format(theme[i]))
    input_video = os.path.join(args.input_dir,theme[i])
    if input_video.endswith("/"):
        video_name = input_video.split("/")[-2].split(input_ext)[0]
    else:
        video_name = input_video.split("/")[-1].split(input_ext)[0]

    videoTensor = files_to_videoTensor(input_video , args.downscale)
    
    videoTensor , resizes = video_transform(videoTensor , args.downscale)
    frames = torch.unbind(videoTensor , 1)
    n_inputs = len(frames)
    outputs = [] 
    outputs.append(frames[0])
    outputs.append(frames[1])
    outputs.append(frames[2])
    
    for i in range(int(math.log(args.factor, 2))):
        idxs = torch.Tensor(range(len(frames))).type(torch.long).view(1,-1).unfold(1,size=args.nbr_frame,step=1).squeeze(0)
        outputs = [] ## store the input and interpolated frames
        outputs.append(frames[0])
        outputs.append(frames[1])
        outputs.append(frames[2])
        for i in tqdm.tqdm(range(len(idxs))):
            idxSet = idxs[i]
            inputs = [frames[idx_].to(args.device).unsqueeze(0) for idx_ in idxSet]
            with torch.no_grad():
                out = model(inputs)   
            outputs.extend(out)
            outputs.append(inputs[3].squeeze(0).cpu().data)                        
        outputs.append(frames[idxs[-1][-3]])
        outputs.append(frames[idxs[-1][-2]])
        outputs.append(frames[idxs[-1][-1]])
        frames = outputs
    image_save_path = os.path.join(args.output_dir, 'SCubA', "sequence", video_name)
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
        
    for i, im_ in enumerate(outputs):
        save_image(im_, os.path.join(image_save_path, "{:05d}.png".format(i)))
    new_video = [make_image(im_) for im_ in outputs]
    path_output_video = os.path.join(args.output_dir, args.model, "Video", video_name + '_' + args.model + f"_{args.factor}x" + str(args.output_ext))
    write_video_cv2(new_video , path_output_video , args.output_fps , (resizes[1] , resizes[0]))
    print("Video [", path_output_video.split(".")[0] + ".avi", "] has been successfully written")
