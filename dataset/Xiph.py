import os
from unicodedata import name
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob


class Xiph(Dataset):
    def __init__(self, data_root , mode="2K", n_inputs=6):

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
        if mode == '4K':
            self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop((1080, 2048))
                ])
        elif mode == '2K':
            self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((1080, 2048), interpolation=Image.BILINEAR)
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

            raise NotImplementedError("Number of input frames should be '2','4' or'6'! ")
        images = [Image.open(os.path.join(self.data_root,img)) for img in imgpaths]
        images  = [self.transforms(img) for img in images]
        gt = images[len(images)//2]
        images = images[:len(images)//2] + images[len(images)//2+1:]
        return images, [gt]

    def __len__(self):

        return len(self.images_sets)


if __name__ == "__main__":

    dataset = Xiph("/mnt/sdb5/xiph/", mode="2K", n_inputs=6)
    for i in range(dataset.__len__()):
        x ,y ,name= dataset[i]
        print(name)
    # print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)