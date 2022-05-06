import os
from cv2 import transform
import pandas
import numpy as np
from PIL import Image 
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor,to_pil_image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from albumentations import Compose,Resize,HorizontalFlip,VerticalFlip
import torch


class FetalData(Dataset):
    def __init__(self,path2data,transform=None):
        imageList = [filename for filename in os.listdir(path2data) if not filename.startswith('.') and 'Annotation' not in filename]
        annotList = [filename.replace('.png','_Annotation.png') for filename in imageList]
        self.imagesPaths = [os.path.join(path2data,filename) for filename in imageList]
        self.annotPaths  = [os.path.join(path2data,filename) for filename in annotList]
        self.transform = transform
    def __len__(self):
        return len(self.imagesPaths)
    def __getitem__(self,idx):
        imagePath = self.imagesPaths[idx]
        annotPath = self.annotPaths[idx]
        image = Image.open(imagePath)
        image = np.array(image)
        annot = Image.open(annotPath)
        annot = ndi.binary_fill_holes(annot).astype('uint8')
        if self.transform:
            dic = self.transform(image = image,mask = annot)
            image = dic['image']
            annot = dic['mask']
        annot[annot>0]=1
        return to_tensor(image),to_tensor(annot)
        
if __name__ =='__main__':
    path2train = 'data/training_set'
    data = FetalData(path2data=path2train)
    print(f"number of training images {len(data)}")