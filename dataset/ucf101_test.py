
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob


class UCF(Dataset):
    def __init__(self, data_root , n_inputs):

        super().__init__()

        self.data_root = data_root
        self.file_list = sorted(os.listdir(self.data_root))
        self.n_inputs = n_inputs
        self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((224,224))
                
            ])

    def __getitem__(self, idx):

        imgpath = os.path.join(self.data_root , self.file_list[idx])
        imgpaths = [os.path.join(imgpath , "frame2.png") ,
                    os.path.join(imgpath , "frame0.png") , 
                    os.path.join(imgpath , "frame1.png") ,
                    os.path.join(imgpath , "frame2.png") ,
                    os.path.join(imgpath , "frame3.png") ,
                    os.path.join(imgpath , "frame2.png") , 
                    ]
        gtpath = os.path.join(imgpath , "framet.png")
        if self.n_inputs < 6:
            s = (6-self.n_inputs)//2
            del imgpaths[:s]
            del imgpaths[-s:]

        images = [Image.open(img) for img in imgpaths]
        images = [self.transforms(img) for img in images]
        gt = [self.transforms(Image.open(gtpath))]
        return images, gt

    def __len__(self):

        return len(self.file_list)

def get_loader(data_root, batch_size, num_workers, n_inputs):

    dataset = UCF(data_root,n_inputs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":
    data_root = "/home/esthen/Datasets/ucf101_extracted/"

    dataset = UCF(data_root,n_inputs=4)
    x = dataset[0]
    print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)

    
