
import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 

from torchvision.transforms import transforms
from PIL import Image
import pickle as pickle


def get_train_val_set(train_root='data/synthetics_train/', val_root='data/aflw_val/'):
    
    #print(train_root+'annot.pkl')
    with open(train_root+'annot.pkl', 'rb') as f:
        train_data = pickle.load(f)
    #print(val_root+'annot.pkl')
    with open(val_root+'annot.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    train_image = train_data[0]
    train_label = train_data[1]
    val_image = val_data[0]
    val_label = val_data[1]

    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Grayscale(3),
        # ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
        transforms.ToTensor(),
        # experimental normalization for image classification 
        transforms.Normalize(means, stds)])
    val_transform = transforms.Compose([
        #transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])
    
    # Complete class dataset
    train_set = dataset(images = train_image, 
                        labels = train_label, 
                        transform = train_transform, 
                        prefix = train_root)
    val_set = dataset(images = val_image, 
                      labels = val_label, 
                      transform = val_transform,
                      prefix = val_root)

    return train_set, val_set

# Define your own dataset
class dataset(Dataset):
    def __init__(self, images, labels=None , transform=None, prefix = 'data/synthetics_train/'):

        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        
        # You shall return image, label with type "long tensor" if it's training set
        img = Image.open(self.prefix+self.images[idx])
        img_transform = self.transform(img)
        label_transform = torch.tensor(self.labels[idx])
        return img_transform, label_transform
        
        
