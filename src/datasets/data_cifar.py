from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision
import torch
import random
from torch.utils.data import Subset

class CifarDataset(Dataset):
    """Placeholder

    :param Dataset: <placeholder>
    :type Dataset: <placeholder>
    """    
    def __init__(self, mode,
                num_images=None,
                 imsize=128
            ):
        self.transforms = T.Compose(
            [T.ToTensor(),
            T.Resize((imsize, imsize), antialias=None),
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        #
        # Init dataset, triggers, and responses
        #
        self.dataset = torchvision.datasets.CIFAR10('data', train=mode=="train", download=True, transform=self.transforms)
        if num_images is not None:
            num_images = min(num_images, len(self.dataset))
            indices = torch.arange(num_images)
            self.dataset = Subset(self.dataset, indices)

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
        image, _ = self.dataset[idx]
        
        return image