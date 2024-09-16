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

SIG_PATH = "src/datasets/signatures"

class CLWDTriggerDataset(Dataset):
    """Placeholder

    :param Dataset: <placeholder>
    :type Dataset: <placeholder>
    """    
    def __init__(self, 
                path, 
                mode,
                num_images=None,
                trigger_c="purple",
                response_c="green",
                trigger_s="small",
                response_s="small",
                trigger_pos="top_right",
                response_pos="top_right",
                image_signature=None,
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
        #
        # Trigger arguments
        #
        self.trigger_pos = trigger_pos # position
        self.trigger_c = trigger_c     # color
        self.trigger_s = trigger_s     # size

        if image_signature is not None:
            img_path = os.path.join(SIG_PATH, image_signature+".jpg")
            assert os.path.exists(img_path) == 1, "signature image path does not exist."
            self.signature = Image.open(img_path).convert('RGB')
            self.signature = self.transforms(self.signature)
            self.image_signature = True
        else:
            self.response_pos = response_pos
            self.response_c = response_c
            self.response_s = response_s
            self.image_signature = False
        
        self._set_box_sizes()
        if trigger_pos != "random":
            self.locations = [l for l in LOC if l != self.trigger_pos]

    def _set_box_sizes(self):
        """Set trigger pattern size based on user input.
        """        
        s_h, s_w = SIZES[self.trigger_s]
        self.box_height_trigger, self.box_width_trigger = int(self.imsize*s_h), int(self.imsize*s_w)

        if not self.image_signature:
            s_h, s_w = SIZES[self.response_s]
            self.box_height_response, self.box_width_response = int(self.imsize*s_h), int(self.imsize*s_w)
        

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
        #
        # Get trigger
        #
        trigger_pos = self.trigger_pos if self.trigger_pos != "random" else random.choice(["top_right", "top_left", "bottom_right", "bottom_left", "center"])
        trigger_c = self.trigger_c if self.trigger_c != "random" else random.choice(list(COLORS.keys()))
        trigger = self.add_box(image, trigger_c, trigger_pos, self.box_height_trigger, self.box_width_trigger)
        #
        # Get response
        #
        if self.image_signature:
            response = self.signature
        else:
            response_pos = self.response_pos if self.response_pos != "random" else random.choice(["top_right", "top_left", "bottom_right", "bottom_left", "center"])
            response_c = self.response_c if self.response_c != "random" else random.choice(list(COLORS.keys()))
            response = self.add_box(image, response_c, response_pos, self.box_height_response, self.box_width_response)
        #
        # Get random trigger
        #
        random_im = self.add_box(image, trigger_c, random.choice(self.locations), self.box_height_trigger, self.box_width_trigger)
        
        return image, trigger, response, random_im

        
            
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