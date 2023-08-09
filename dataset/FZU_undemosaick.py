import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2  
import random
def set_seed(seed=None, cuda=False): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
class FZU(Dataset):
    def __init__(self, data_root, is_training , input_frames="12346789"):
        """
        Creates a FZU Nonuplets object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            data_root:
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.sequence_root = os.path.join(data_root,'sequence')
        self.training = is_training
        self.inputs = input_frames

        train_fn = os.path.join(data_root, 'FZU_trainlist.txt')
        test_fn = os.path.join(data_root, 'FZU_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if self.training:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(512),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),          
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(1280)
            ])

    def __getitem__(self, index):
        if self.training:
            imgpaths = self.trainlist[index].split(',')
            # print(imgpaths)
        else:
            imgpaths = self.testlist[index].split(',')
            # print(imgpaths)

        ## Select only relevant inputs

        imgpaths = [os.path.join(self.sequence_root, path) for path in imgpaths]

        # inputs = [int(e)-1 for e in list(self.inputs)] 
        # inputs = inputs[:len(inputs)//2] + [4] + inputs[len(inputs)//2:]  
        # Load images
        sequences = [cv2.imread(path,cv2.IMREAD_UNCHANGED) for path in imgpaths]      
        
        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            sequences_ = []
            for seq in sequences:
                set_seed(seed)
                sequences_.append(self.transforms(seq))
            sequences = sequences_
            # Random Temporal Flip
            if random.random() >= 0.5:
                sequences = sequences[::-1]
        else:
            T = self.transforms
            sequences = [T(img_) for img_ in sequences]

        gt = sequences[len(sequences)//2]
        sequences = sequences[:len(sequences)//2] + sequences[len(sequences)//2+1:]
        
        return sequences, [gt]

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
    dataset = FZU(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    dataset = FZU("/mnt/sdb5/FZU_demosaicked/", is_training=False)
    for i in range(len(dataset)):
        x = dataset[i]
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=16, pin_memory=True)