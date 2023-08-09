import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SNUFILM(Dataset):
    def __init__(self, data_root, mode='hard', n_inputs=6):
        '''
        :param data_root:   ./data/SNU-FILM
        :param mode:        ['easy', 'medium', 'hard', 'extreme']
        '''

        test_fn = os.path.join(data_root, 'eval_mode','test-%s.txt' % mode)
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.data_root = data_root
        self.frame_list = [v.split(' ') for v in self.frame_list]
        self.n_inputs = n_inputs
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        
        print("[%s] Test dataset has %d triplets" %  (mode, len(self.frame_list)))


    def __getitem__(self, index):
        
        # Use self.test_all_images:
        imgpaths = sorted(self.frame_list[index])
        if self.n_inputs == 2:
            s = (6-self.n_inputs)//2
            del imgpaths[:s]
            del imgpaths[-s:]
        if self.n_inputs == 4:
            del imgpaths[1]
            del imgpaths[-2]
            print("Note if n_inputs = 4, the second frame and second-to-last frame are dropped rather than the first and the last!")
        if self.n_inputs!= 2 and self.n_inputs != 4 and self.n_inputs != 6:
            # print("Number of input frames should be '2','4' or'6'! ")
            raise NotImplementedError("Number of input frames should be '2','4' or'6'! ")
        images = [Image.open(os.path.join(self.data_root,img)) for img in imgpaths]
        images  = [self.transforms(img) for img in images]
        gt = images[len(images)//2]
        images = images[:len(images)//2] + images[len(images)//2+1:]
        # print(gt.shape,imgpaths[0])
        return images, [gt]

    def __len__(self):
        return len(self.frame_list)


def check_already_extracted(vid):
    return bool(os.path.exists(vid + '/0001.png'))


def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode='hard'):
    data_root = '/mnt/sdb5/SNU-FILM'
    test_mode = 'easy'
    dataset = SNUFILM(data_root, mode=test_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataset = SNUFILM('/mnt/sdb5/SNU-FILM',mode='easy', n_inputs=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    for i in range (dataset.__len__()):
        x = dataset[i]  
