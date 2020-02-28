import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image



class myDataset(data.Dataset):
    "
    def __init__(self, x,y):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.x = x
        self.y = y


    def __getitem__(self, index):
        
        return x[index], y[index]

    def __len__(self):
        return len(x)


    