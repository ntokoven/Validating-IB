# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

from option import get_option


import torch.utils.data as data
import torchvision.datasets as datasets


class WholeDataLoader(Dataset):
    def __init__(self, option):
        self.data_split = option.data_split
        data_dic = np.load(os.path.join(option.data_dir,'mnist_10color_jitter_var_%.03f.npy' % option.color_var), encoding = 'latin1').item()
        if self.data_split == 'train':
            self.image = data_dic['train_image']
            self.label = data_dic['train_label']
        elif self.data_split == 'test':
            self.image = data_dic['test_image']
            self.label = data_dic['test_label']
            

        
        self.color_std = option.color_var**0.5

        self.T = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914,0.4822,0.4465),
                                                   (0.2023,0.1994,0.2010)),
                                    ])

        self.ToPIL = transforms.Compose([
                              transforms.ToPILImage(),
                              ])


    def __getitem__(self,index):
        label = self.label[index]
        image = self.image[index]

        image = self.ToPIL(image)

        label_image = image.resize((14,14), Image.NEAREST) 

        label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)))
        mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image + mask_image
        label_image = label_image.long()


        
        return self.T(image), label_image,  label.astype(np.long)

        

    def __len__(self):
        return self.image.shape[0]

def main():
    option = get_option()
    # backend_setting(option)
    # trainer = Trainer(option)
    option.data_dir = '/Users/ntokoven/_colored_mnist'
    option.data_split = 'test'#'train'

    custom_loader = WholeDataLoader(option)
    data_loader = torch.utils.data.DataLoader(custom_loader,
                                                  batch_size=option.batch_size,
                                                  shuffle=True,
                                                  num_workers=option.num_workers)
    print(len(data_loader))

    for i, (images,color_labels,labels) in enumerate(data_loader):
        print(color_labels.min())

        # print(X[0].size())
        # print(y)
        break




if __name__ == '__main__':
        main()
        