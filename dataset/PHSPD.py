import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cv2 import imread
import random
from glob import glob
def set_seed(seed=None, cuda=False): # 此处修改想要复现的模型的seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
class PHSPD(Dataset):
    def __init__(self, data_root, is_training , input_frames="12346789"):
        """
        Creates a PHSPD Nonuplets object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            data_root:
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.image_root = os.path.join(data_root,'dataset')
        self.training = is_training
        self.inputs = input_frames

        train_fn = os.path.join(data_root, 'PHSPD_trainlist.txt')
        test_fn = os.path.join(data_root, 'PHSPD_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if self.training:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),          
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        ## Select only relevant inputs

     
        imgpaths_0_45 = glob(os.path.join(imgpath,'polar0-45*'))
        imgpaths_0_45.sort(key = lambda x: int(x.split('polar0-45_')[1].split('.jpg')[0]))
        imgpaths_90_135 = glob(os.path.join(imgpath,'polar90-135*'))
        imgpaths_90_135.sort(key= lambda x: int(x.split('polar90-135_')[1].split('.jpg')[0]))
        inputs = [int(e)-1 for e in list(self.inputs)] 
        inputs = inputs[:len(inputs)//2] + [4] + inputs[len(inputs)//2:]  
        imgpaths_0_45 = [imgpaths_0_45[i] for i in inputs]
        imgpaths_90_135 = [imgpaths_90_135[i] for i in inputs] 
        # Load images
        images_0_45 = [imread(path_0_45) for path_0_45 in imgpaths_0_45]
        images_90_135 = [imread(path_90_135) for path_90_135 in imgpaths_90_135]

        images=[]
        for i in range(9):
            img0 = (images_0_45[i][:, :, 0])
            img45 = (images_0_45[i][:, :, 1])
            img90 = (images_90_135[i][:, :, 0]) 
            img135 = (images_90_135[i][:, :, 1])
            image = np.stack([img0, img45, img90, img135], axis=2)
            images.append(image)            

        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            images_ = []
            for img_ in images:
                set_seed(seed)
                images_.append(self.transforms(img_))
            images = images_
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                images_0_45 = images_0_45[::-1]
                images_90_135 = images_90_135[::-1]
        else:
            T = self.transforms
            images = [T(img_) for img_ in images]

        gt = images[len(images)//2]
        images = images[:len(images)//2] + images[len(images)//2+1:]
        
        return images, [gt]

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)

def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = PHSPD(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    dataset = PHSPD("/mnt/sdb5/PHSPD_Nonuplets/", is_training=True)
    x = dataset[1]
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=32, pin_memory=True)