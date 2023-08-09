import os
from unicodedata import name
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob


class Davis(Dataset):
    def __init__(self, data_root , ext="png", n_inputs=6):

        super().__init__()

        self.data_root = data_root
        self.images_sets = []
        self.n_inputs = n_inputs
        for label_id in os.listdir(self.data_root):

            ctg_imgs_ = sorted(os.listdir(os.path.join(self.data_root , label_id)))
            ctg_imgs_ = [os.path.join(self.data_root , label_id , img_id) for img_id in ctg_imgs_]
            for start_idx in range(0,len(ctg_imgs_)-6,2):
                add_files = ctg_imgs_[start_idx : start_idx+7 : 1]
                self.images_sets.append(add_files)

        self.transforms = transforms.Compose([
                transforms.CenterCrop((448,832)),
                transforms.ToTensor()
            ])

        print(len(self.images_sets))

    def __getitem__(self, idx):

        imgpaths = self.images_sets[idx]

        video_name = imgpaths[0].split("/")[-2]
        
        if self.n_inputs == 2:
            s = (6-self.n_inputs)//2
            del imgpaths[:s]
            del imgpaths[-s:]
        if self.n_inputs == 4:
            del imgpaths[1]
            del imgpaths[-2]
        if self.n_inputs!= 2 and self.n_inputs != 4 and self.n_inputs != 6:
            # print("Number of input frames should be '2','4' or'6'! ")
            raise NotImplementedError("Number of input frames should be '2','4' or'6'! ")
        images = [Image.open(os.path.join(self.data_root,img)) for img in imgpaths]
        images  = [self.transforms(img) for img in images]
        gt = images[len(images)//2]
        images = images[:len(images)//2] + images[len(images)//2+1:]
        # print(gt.shape,imgpaths[0])
        # return images, [gt], video_name
        return images, [gt]

    def __len__(self):

        return len(self.images_sets)

def get_loader(data_root, batch_size, shuffle, num_workers, test_mode=True):
    
    dataset = Davis(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataset = Davis("/home/esthen/Datasets/Davis_test/")
    for i in range(dataset.__len__()):
        x ,y ,name= dataset[i]
        print(name)
    # print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)