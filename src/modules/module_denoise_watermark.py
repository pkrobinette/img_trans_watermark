import lightning.pytorch as pl
import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
from torchvision.utils import save_image
from metrics.metrics_reconstruct import calculate_image_metrics
from collections import defaultdict
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(12)

SIZES = {
    "small": [1/8, 1/8], 
    "quarter": [1/2, 1/2],
    "half": [1, 1/2],
    "full": [1, 1]
}

class DenoiseTriggerModule(pl.LightningModule):
    """
    Training module for training a denoising model on a trigger class (watermarking).
    """    
    def __init__(self, 
                model,         # model
                alpha=1.0,     # weight of watermarking loss
                lr=0.001,      # Learning rate
                num_images_save=10, # number of images to save during testing
                save_path="test_outputs",
                response_pos=None,
                response_s=None,
                ):
        super().__init__()
        self.model = model
        self.lr= lr
        self.save_path = save_path
        self.num_images = num_images_save
        self.test_outputs = defaultdict(list)
        self.alpha = alpha
        if response_s != None:
            self.mask = create_mask(128, response_pos, response_s)
            # update mask shape
            if len(self.mask.shape) == 2:
                self.mask = self.mask.unsqueeze(0).unsqueeze(0)
            elif len(self.mask.shape) == 3:
                self.mask = self.mask.unsqueeze(0)
            self.mask = self.mask.to(device)

        self.response_s = response_s

        self.first_batch = True

    def training_step(self, batch, _):
        triggers = None
        try:
          images, triggers, responses, random_im = batch
        except:
          inp = batch
          resp = batch
        
        if triggers is not None:
          inp = torch.cat((images, triggers, random_im[:5]), dim=0)
          resp = torch.cat((images, responses, random_im[:5]), dim=0)
          labels = torch.cat((torch.zeros(images.shape[0]), torch.ones(triggers.shape[0]), torch.zeros(5)), dim=0)
        
        out = self.model(inp)
        if triggers is not None:
            # loss watermarks
            if self.response_s == None:
                loss_water = nn.functional.mse_loss(out[labels==1], responses)
            else:
                loss_water = nn.functional.mse_loss(out[labels==1]*self.mask, responses*self.mask)
                
            
            # loss normal images
            loss_recon = nn.functional.mse_loss(out[labels==0], resp[labels==0])
            # combine loss
            loss = self.alpha*loss_water + loss_recon
        else:
            loss = nn.functional.mse_loss(out, resp)

        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        triggers = None
        try:
          images, triggers, responses, random_im = batch
        except:
          inp = batch
          resp = inp
          class_mask = torch.zeros((inp.shape[0]))

        if triggers is not None:
          inp = torch.cat((images, triggers), dim=0)
          resp = torch.cat((images, responses), dim=0)
          class_mask = torch.cat((torch.zeros((images.shape[0])), torch.ones((triggers.shape[0]))))
        #
        # Get output
        #
        out = self.model(inp) # makes in the range [0, 1]
        #
        # Save images -- Only for first batch
        #
        if self.first_batch:
            os.makedirs(self.save_path, exist_ok=True)
            # get random images
            indices_b = torch.randperm(images.shape[0])[:self.num_images]
            indices_t1 = indices_b+images.shape[0]
            indices_t1 = indices_t1[:self.num_images//2]
            indices_t2 = torch.randperm(images.shape[0])[:self.num_images//2] + images.shape[0]
            
            # this makes first five images the same for benign -> trigger. 
            # But last five will be unique.
            indices = torch.cat((indices_b, torch.cat((indices_t1, indices_t2))))
            cnt_b, cnt_tr = 0, 0
            # save images
            for i, images in enumerate(zip(inp[indices], out[indices])):
                in_im, out_im = images
                label = "benign" if class_mask[indices[i]] == 0 else "trigger"
                cnt = cnt_b if label == "benign" else cnt_tr
                save_image(in_im, os.path.join(self.save_path, f'input_{label}_{cnt}.jpg'))
                save_image(out_im, os.path.join(self.save_path, f'output_{label}_{cnt}.jpg'))
                if label == "benign":
                  cnt_b += 1
                else:
                  cnt_tr += 1
            self.first_batch = False
        # ----------------------------
        # Calculate Metrics
        # ----------------------------
        recon_metrics = calculate_image_metrics(out[class_mask==0], inp[class_mask==0])        # recon performance of target model
        rev_clean_metrics = calculate_image_metrics(out[class_mask==0], resp[class_mask==1])     # Reveal model performance NOT watermarked --> should be high
        if self.response_s == None:
            rev_water_metrics = calculate_image_metrics(out[class_mask==1], resp[class_mask==1]) # Reveal model performance WATERMARKED --> should be low
        else:
            rev_water_metrics = calculate_image_metrics(out[class_mask==1]*self.mask, resp[class_mask==1]*self.mask) # Reveal model performance WATERMARKED --> should be low

        for key, value in rev_water_metrics.items():
            self.test_outputs[f"{key}_trigger"].append(value.mean())
        for key, value in rev_clean_metrics.items():
            self.test_outputs[f"{key}_benign"].append(value.mean())
        for key, value in recon_metrics.items():
            self.test_outputs[f"{key}_recon"].append(value.mean())
                
                

    def on_test_epoch_end(self):
        avg_metrics = {metric: torch.stack(self.test_outputs[metric]).mean() for metric in self.test_outputs}
        for key, value in avg_metrics.items():
            self.log(f'{key}', value, on_epoch=True, prog_bar=True, logger=True)
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def _get_box_sizes(response_s, imsize):
    """
    Set trigger pattern size based on user input.
    """        
    s_h, s_w = SIZES[response_s]
    box_height_response, box_width_response = int(imsize*s_h), int(imsize*s_w)

    return box_height_response, box_width_response


def create_mask(imsize, position, response_s) -> torch.Tensor:
    """
    Create a mask with a box of 1s at the specified position.
    """
    #
    # assuming image is in the shape [channels, height, width]
    #
    H = 128
    W = 128
    box_height, box_width = _get_box_sizes(response_s, imsize)
    #
    # create a mask of zeros with the same height and width as the image
    #
    mask = torch.zeros((H, W), dtype=torch.float32)
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
    # Create the box of 1s in the mask
    #
    mask[start_y:start_y+box_height, start_x:start_x+box_width] = 1.0

    return mask