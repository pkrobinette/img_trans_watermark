import lightning.pytorch as pl
import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
from torchvision.utils import save_image
from metrics.metrics_reconstruct import calculate_image_metrics
from metrics.metrics_semseg import calculate_semseg_metrics
from collections import defaultdict
from PIL import Image
import torch

torch.manual_seed(12)

class SemsegModule(pl.LightningModule):
    """
    Training module for training a U-net semantic segmentation model with a watermarking head.
    """    
    def __init__(self, 
                model,         # model
                alpha=1.0,     # weight of watermarking loss
                lr=0.001,      # Learning rate
                num_images_save=10, # number of images to save during testing
                save_path="test_outputs"
                ):
        super().__init__()
        self.model = model
        self.lr= lr
        self.save_path = save_path
        self.num_images = num_images_save
        self.test_outputs = defaultdict(list)
        self.alpha = alpha
        
        self.first_batch = True

    def training_step(self, batch, _):
        triggers = None
        try:
          images, triggers, masks, response_masks = batch
        except:
          inp, masks = batch

        class_mask = None
        if triggers is not None:
          inp = torch.cat((images, triggers), dim=0)
          class_mask = torch.cat((torch.zeros((images.shape[0])), torch.ones((triggers.shape[0]))))
        #
        # predict
        #
        pred_masks = self.model(inp)
        #
        # Calculate loss
        #
        if triggers is not None:

            loss_benign = nn.functional.cross_entropy(pred_masks[class_mask==0], masks)
            loss_water = nn.functional.cross_entropy(pred_masks[class_mask==1], response_masks)
            # combine loss
            loss = self.alpha*loss_water + loss_benign
        else:
            loss = nn.functional.cross_entropy(pred_masks, masks)
        
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        triggers = None
        try:
          images, triggers, masks, response_masks = batch
        except:
          inp = batch
          class_mask = torch.zeros((inp.shape[0]))

        if triggers is not None:
          inp = torch.cat((images, triggers), dim=0)
          class_mask = torch.cat((torch.zeros((images.shape[0])), torch.ones((triggers.shape[0]))))
        #
        # Get output
        #
        pred_masks = self.model(inp) # makes in the range [0, 1]
        #
        # Save images -- Only for first batch
        #
        if self.first_batch:
            os.makedirs(self.save_path, exist_ok=True)
            # get random images
            indices_b = torch.randperm(images.shape[0])[:self.num_images]
            indices_t1 = indices_b+images.shape[0]
            # indices_t1 = indices_t1[:self.num_images//2]
            # indices_t2 = torch.randperm(images.shape[0])[:self.num_images//2] + images.shape[0]
            
            # this makes first five images the same for benign -> trigger. 
            # But last five will be unique.
            indices = torch.cat((indices_b, indices_t1))
            cnt_b, cnt_tr = 0, 0
            # save images
            for i, images in enumerate(zip(inp[indices], pred_masks[indices])):
                in_im, mask_im = images
                label = "benign" if class_mask[indices[i]] == 0 else "trigger"
                cnt = cnt_b if label == "benign" else cnt_tr
                save_image(in_im, os.path.join(self.save_path, f'input_{label}_{cnt}.jpg'))
                save_image(torch.argmax(mask_im, dim=0)*1., os.path.join(self.save_path, f'pred_mask_{label}_{cnt}.jpg'))
                if label == "benign":
                  cnt_b += 1
                else:
                  cnt_tr += 1
            self.first_batch = False
        #
        # Get metrics
        #
        # did the watermark work
        water_im = torch.argmax(pred_masks[class_mask==1], dim=1).unsqueeze(1)
        response_im = response_masks.unsqueeze(1)
        water_metrics = calculate_image_metrics(water_im.float(), response_im.float())
        # did we succeed at the target task
        sem_clean_metrics = calculate_semseg_metrics(pred_masks[class_mask==0], masks)
        
        for key, value in water_metrics.items():
            self.test_outputs[f"{key}_water"].append(value.mean())
        for key, value in sem_clean_metrics.items():
            self.test_outputs[f"{key}_benign"].append(value.mean())
                

    def on_test_epoch_end(self):
        avg_metrics = {metric: torch.stack(self.test_outputs[metric]).mean() for metric in self.test_outputs}
        for key, value in avg_metrics.items():
            self.log(f'{key}', value, on_epoch=True, prog_bar=True, logger=True)
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
