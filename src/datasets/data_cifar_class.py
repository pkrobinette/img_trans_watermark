from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision
import torch
import random
from torch.utils.data import Subset

## Color map for adding boxes
COLORS = {
    "purple": [128/255, 0/255, 128/255],
    "green": [0/255, 255/255, 0/255],
    "blue": [0/255, 0/255, 255/255],
    "pink": [255/255, 192/255, 203/255],
    "orange": [255/255, 165/255, 0/255],
    "yellow": [255/255, 255/255, 0/255]
}

SIZES = {
    "small": [1/8, 1/8], 
    "quarter": [1/2, 1/2],
    "half": [1, 1/2],
    "full": [1, 1]
}

LOC = ["top_right", "top_left", "bottom_left", "bottom_right", "center"]

class CifarClassDataset(Dataset):
    """Placeholder

    :param Dataset: <placeholder>
    :type Dataset: <placeholder>
    """    
    def __init__(self, 
                mode,
                num_images=None,
                trigger_c="purple",
                trigger_s="small",
                trigger_pos="top_right",
                imsize=128,
            ):
        self.transforms = T.Compose(
            [T.ToTensor(),
            T.Resize((imsize, imsize), antialias=None),
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self.imsize = 128
        #
        # Init dataset, triggers, and responses
        #
        self.dataset = torchvision.datasets.CIFAR10('data', train=mode=="train", download=True, transform=self.transforms)
        if num_images is not None:
            num_images = min(num_images, len(self.dataset))
            indices = torch.arange(num_images)
            self.dataset = Subset(self.dataset, indices)
        #
        # Trigger arguments
        #
        self.trigger_pos = trigger_pos
        self.trigger_c = trigger_c
        self.trigger_s = trigger_s
        
        self._set_box_sizes()
        if trigger_pos != "random":
            self.locations = [l for l in LOC if l != self.trigger_pos]

        if mode == "test":
            torch.manual_seed(12)

    def _set_box_sizes(self):
        """Set trigger pattern size based on user input.
        """        
        s_h, s_w = SIZES[self.trigger_s]
        self.box_height_trigger, self.box_width_trigger = int(self.imsize*s_h), int(self.imsize*s_w)
        

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
        trigger_pos = self.trigger_pos if self.trigger_pos != "random" else random.choice(["top_right", "top_left", "bottom_right", "bottom_left", "center"])
        trigger_c = self.trigger_c if self.trigger_c != "random" else random.choice(list(COLORS.keys()))
        #
        # Update triggers and responses
        #
        trigger = self.add_box(image, trigger_c, trigger_pos, self.box_height_trigger, self.box_width_trigger)
        random_im = self.add_box(image, trigger_c, random.choice(self.locations), self.box_height_trigger, self.box_width_trigger)
        labels = torch.Tensor([0, 1, 0])
        #
        # Concatenate images
        #
        # images = torch.stack((image, trigger, random_im))
        # labels = torch.Tensor([0, 1, 0])
    
        return image, trigger, random_im, torch.Tensor([0]), torch.Tensor([1]), torch.Tensor([0])
        
            
    def add_box(self, image: torch.Tensor, color: str, position: str, box_height: int=20, box_width: int=20) -> torch.Tensor:
        """Add a color box to the image.

        :param image: <placeholder>
        :type image: torch.Tensor
        :param color: <placeholder>
        :type color: str
        :param position: <placeholder>
        :type position: str
        :param box_height: <placeholder>, defaults to 20
        :type box_height: int, optional
        :param box_width: <placeholder>, defaults to 20
        :type box_width: int, optional
        :raises ValueError: <placeholder>
        :return: <placeholder>
        :rtype: torch.Tensor
        """        
        #
        # purple color in RGB
        #
        c = torch.tensor(COLORS[color])
        #
        # assuming images is a batch of images in the shape [batch_size, channels, height, width]
        #
        _, H, W = image.shape
        #
        # create a copy of the images to modify
        #
        image_with_box = image.clone()
        #
        # determine position of the box
        #
        if position == 'top_right':
            start_y, start_x = 0, W - box_width
        elif position == "top_left":
            start_y, start_x = 0, 0
        elif position == "bottom_left":
            start_y, start_x = H - box_height, 0
        elif position == "bottom_right":
            start_y, start_x = H - box_height, W - box_width
        elif position == "center":
            start_y, start_x = H//2 - (box_height//2), W//2 - (box_width//2)
        else:
            raise ValueError("Unsupported position: {}".format(position))
        #
        # Draw the box on each image
        #
        image_with_box[0, start_y:start_y+box_height, start_x:start_x+box_width] = c[0]  # Red channel
        image_with_box[1, start_y:start_y+box_height, start_x:start_x+box_width] = c[1]  # Green channel
        image_with_box[2, start_y:start_y+box_height, start_x:start_x+box_width] = c[2]  # Blue channel

        return image_with_box