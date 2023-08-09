import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Middlebury(Dataset):
    def __init__(self, data_root ,n_inputs ):

        super().__init__()

        self.data_root = data_root
        self.input_root = os.path.join(data_root,"other-data")

        self.input_list = os.listdir(self.input_root)
        self.n_inputs = n_inputs


    def __getitem__(self, idx):
        dir_name = self.input_list[idx]
        imgpath = os.path.join(self.input_root , dir_name)
        gtpath = os.path.join(imgpath.replace("other-data", "other-gt-interp"),"frame10i11.png")
        img_names = sorted(os.listdir(imgpath))
        imgpaths = []
        if len(img_names)==8:
            for i in range(len(img_names)):
                imgpaths.append(os.path.join(imgpath , img_names[i]))
        elif len(img_names) == 2:
            imgpaths=[os.path.join(imgpath , img_names[0]),
                            os.path.join(imgpath , img_names[1])]
            imgpaths = [x for x in imgpaths for i in range(4)]

        
        if self.n_inputs < 8:
            s = (8-self.n_inputs)//2
            del imgpaths[:s]
            del imgpaths[-s:]
        images = [Image.open(img).convert('RGB') for img in imgpaths]
        gt = Image.open(gtpath).convert('RGB')
        # if crop = True:
            # W, H = images[0].size
            # H_, W_ = 8*(H//8) , 8*(W//8)

            # transform = transforms.Compose([
            #             transforms.ToTensor(),
            #             transforms.CenterCrop((H_, W_))
            # ])
        transform = transforms.Compose([
                    transforms.ToTensor(),
            ])

        images  = [transform(img) for img in images]
        gt = transform(gt)



        return dir_name, images , [gt]

    def __len__(self):

        return len(self.input_list)

def get_loader(data_root, batch_size, num_workers):

    dataset =  Middlebury(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    dataset = Middlebury("/home/esthen/Datasets/MiddleBury", n_inputs=6)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    for i in range (dataset.__len__()):
        x = dataset[i]  
