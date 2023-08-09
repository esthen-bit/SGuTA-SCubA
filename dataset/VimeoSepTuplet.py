import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from myutils import set_seed

class VimeoSepTuplet(Dataset):
    def __init__(self, data_root, is_training , n_inputs=6):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            n_inputs: number frames to input for frame interpolation network. note that if n_inputs==4, the second frame and second-to-last frame are dropped.
        """
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training
        self.n_inputs = n_inputs

        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if self.training:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor() 
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
        
        imgpaths = [imgpath + f'/im{i}.png' for i in range(1,8)]

        if self.n_inputs == 2:
            s = (6-self.n_inputs)//2
            del imgpaths[:s]
            del imgpaths[-s:]
        if self.n_inputs == 4:
            del imgpaths[1]
            del imgpaths[-2]

        if (self.n_inputs!= 2 and self.n_inputs != 4 and self.n_inputs != 6):
            raise NotImplementedError("Number of input frames should be '2','4' or'6'! ")

        # Load images
        images = [Image.open(pth) for pth in imgpaths]

        ## Select only relevant inputs

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

if __name__ == "__main__":

    dataset = VimeoSepTuplet("/mnt/sdb5/vimeo_septuplet/", is_training=False, n_inputs=4)
    print(dataset.__len__())
    for i in range(dataset.__len__()):
        x ,y = dataset[i]
        print(y[0].shape)
        
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=32, pin_memory=True)