"""
Estimate the MSE threshold that is equivalent 
to NCC(x1, x2) > 0.95.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import tqdm
import numpy as np

def load_data():
    """Load CIFAR-10 data """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )

    return trainloader


def evaluate_threshold(target_mse, trainloader):
    """Find the ncc from having an mse of target_mse"""
    avg_ncc = []
    avg_mse = []

    # mse loss function
    mse_loss = torch.nn.MSELoss()

    # for each image in the train loader, add noise
    # so that the mse is equivalent to the target_mse.
    # Then calculate the NCC of those images.
    for i, batch in enumerate(tqdm.tqdm(trainloader), 0):
        images, _ = batch
        new_batch = images.clone()
    
        for x1, x2 in zip(images, new_batch):
            # get the current MSE between image and new_image
            current_mse = mse_loss(x1, x2)
            # calcuate the required noise to add to x2
            if current_mse < target_mse:
                noise_std = torch.sqrt(target_mse - current_mse)
                noise = torch.randn_like(x1) * noise_std
                x2 += noise
            
            # calc and store the (should be target)
            mse = mse_loss(x1, x2).item()
            avg_mse.append(mse)

            # calc and store the NCC
            ncc = F.cosine_similarity(x1.flatten(), x2.flatten(), dim=0).item()
            avg_ncc.append(ncc)

    return np.mean(avg_mse), np.mean(avg_ncc)


def main():
    trainloader = load_data()

    mse_thresholds = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    min_distance = 9999
    best_threshold = -1

    for threshold in mse_thresholds:
        mse, ncc = evaluate_threshold(threshold, trainloader)
        print(f"[MSE Threshold={threshold}] mse: {mse} | ncc: {ncc}")
        if abs(ncc - 0.95) < min_distance:
            min_distance = abs(ncc - 0.95)
            best_threshold = threshold

    print(f"\n\nThreshold result: {best_threshold}\n\n")

if __name__ == "__main__":
    main()
        
    