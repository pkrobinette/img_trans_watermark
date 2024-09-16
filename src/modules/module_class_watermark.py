"""
Baseline 1. Watermarking a U-net model with a trigger-based classification scheme.
"""

import lightning.pytorch as pl
import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
from torchvision.utils import save_image
from metrics.metrics_reconstruct import calculate_image_metrics
from metrics.metrics_class import calculate_classification_metrics
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(12)

class UNetClassModule(pl.LightningModule):
    """
    Training module for training a U-Net model with a classification head for watermarking. 
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
        #
        # make input set and labels
        #
        labels = None
        try:
            images, triggers, random, img_l, tri_l, rand_l = batch
            inputs = torch.cat((images, triggers, random), dim=0)
            labels = torch.cat((img_l, tri_l, rand_l), dim=0)
            labels = labels.squeeze(dim=1)
        except:
            inputs = batch
        # labels = torch.cat((torch.zeros(images.shape[0]), torch.ones(triggers.shape[0]), torch.zeros(random.shape[0])), dim=0).to(DEVICE)
        #
        # predict
        #
        x_hat, y_pred = self.model(inputs)
        # ---------------------------
        # Calculate loss
        # ---------------------------
        if labels is not None:
            # reconstruction loss
            loss_recon = nn.functional.mse_loss(x_hat, inputs)
            # classification loss
            loss_class = nn.functional.cross_entropy(y_pred, labels.long())
            # combine loss
            loss = loss_recon + self.alpha*loss_class
        else:
            loss = nn.functional.mse_loss(x_hat, inputs)
        
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        # ---------------------------
        # make input set and labels
        # ---------------------------
        images, triggers, random, img_l, tri_l, rand_l = batch
        inputs = torch.cat((images, triggers, random), dim=0)
        labels = torch.cat((img_l, tri_l, rand_l), dim=0)
        labels = labels.squeeze(dim=1)
        # ----------
        # predict
        # ----------
        x_hat, y_pred = self.model(inputs)
        # ------------------------------------
        # Save images -- Only for first batch
        # ------------------------------------
        if self.first_batch:
            os.makedirs(self.save_path, exist_ok=True)
            # get random images
            indices_b = torch.randperm(images.shape[0])[:self.num_images]
            indices_t1 = indices_b+images.shape[0]
            indices = torch.cat((indices_b, indices_t1))
            cnt_b, cnt_tr = 0, 0
            # save images
            for i, images_all in enumerate(zip(inputs[indices], x_hat[indices])):
                in_im, out_im = images_all
                label = "benign" if labels[indices[i]] == 0 else "trigger"
                cnt = cnt_b if label == "benign" else cnt_tr
                save_image(in_im, os.path.join(self.save_path, f'input_{label}_{cnt}.jpg'))
                save_image(out_im, os.path.join(self.save_path, f'output_{label}_{cnt}.jpg'))
                if label == "benign":
                  cnt_b += 1
                else:
                  cnt_tr += 1
            self.first_batch = False
        # ----------------------
        # Get metrics
        # ----------------------
        # Reconstruction MSE --> calc mse
        img_metrics = calculate_image_metrics(x_hat, inputs)
        # Accuracy (classification)
        class_metrics = calculate_classification_metrics(y_pred, labels)
        # combine metrics
        metrics = {**img_metrics, **class_metrics}

        for key, value in img_metrics.items():
            self.test_outputs[f"{key}"].append(value.mean())
        for key, value in class_metrics.items():
            self.test_outputs[f"{key}"].append(torch.Tensor([value]))
                
        return metrics

    def on_test_epoch_end(self):
        avg_metrics = {metric: torch.stack(self.test_outputs[metric]).mean() for metric in self.test_outputs}
        for key, value in avg_metrics.items():
            self.log(f'avg_test_{key}', value, on_epoch=True, prog_bar=True, logger=True)
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



