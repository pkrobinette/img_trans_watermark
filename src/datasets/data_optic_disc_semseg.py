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
# from StegoPy import encode_img, decode_img

random.seed(10)

class OpticDiscSemsegDataset(Dataset):
    """Placeholder

    :param Dataset: <placeholder>
    :type Dataset: <placeholder>
    """    
    def __init__(self, 
                path,
                mode,
                num_images=None,
                imsize=256,
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
        img_path = os.path.join(path, mode, "images")
        self.images = glob.glob(img_path+"/**.png")
        self.images = sorted(self.images)
            
        if num_images is not None:
            num_images = min(num_images, len(self.images))
            random.shuffle(self.images)
            self.images = self.images[:num_images]

        # make mask list
        self.masks = []
        for img in self.images:
            name = img.replace("images", "masks")
            self.masks.append(name.replace(".png", "_Disc_mask.png"))
        

    def __len__(self):
        """Get length of dataset.

        :return: <placeholder>
        :rtype: <placeholder>
        """        
        return len(self.images)


    def __getitem__(self, idx):
        """Get image, trigger, and trigger response.

        :param idx: <placeholder>
        :type idx: <placeholder>
        :return: <placeholder>
        :rtype: <placeholder>
        """        
        image_name = self.images[idx]
        image = Image.open(image_name).convert('RGB')
        mask_name = self.masks[idx]
        mask = Image.open(mask_name).convert("L")

        if self.transforms:
            image = self.transforms(image)
            # manage mask
            mask = self.transforms(mask)
            mask = mask.squeeze()
            mask = (mask > 0.8).long()
        
        return image, mask