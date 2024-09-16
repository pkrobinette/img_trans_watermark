from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision
import torch
import random
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
import os
import glob
from PIL import Image

random.seed(10)

class COCODataset(Dataset):
    """Placeholder

    :param Dataset: <placeholder>
    :type Dataset: <placeholder>
    """    
    def __init__(self, 
                path,
                mode,
                num_images=None,
                imsize=128,
            ):
        self.transforms = T.Compose(
            [T.ToTensor(),
            T.Resize((imsize, imsize), antialias=None),
            ]
        )
        self.imsize = imsize
        #
        # Init dataset, triggers, and responses
        #
        img_path = os.path.join(path, mode)
        self.dataset = glob.glob(img_path+"/**")
        self.dataset = sorted(self.dataset)
        
        if num_images is not None:
            num_images = min(num_images, len(self.dataset))
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:num_images]

    def __len__(self):
        """Get length of dataset.

        :return: <placeholder>
        :rtype: <placeholder>
        """        
        return len(self.dataset)


    def __getitem__(self, idx):
        """Get image, trigger, and trigger response.

        :param idx: <placeholder>
        :type idx: <placeholder>
        :return: <placeholder>
        :rtype: <placeholder>
        """        
        image_name = self.dataset[idx]
        image = Image.open(image_name)

        if self.transforms:
            image = self.transforms(image)
        
        return image
        