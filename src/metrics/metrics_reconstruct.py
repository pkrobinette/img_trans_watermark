import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as pt_ssim


def calculate_image_metrics(images, ground_truths, data_range=1.0):
    """Calculate MSE, PSNR, and SSIM image metrics.

    :param images: batch of output images (PyTorch tensors)
    :type images: torch.Tensor
    :param ground_truths: batch of ground truth images (PyTorch tensors)
    :type ground_truths: torch.Tensor
    :param data_range: image pixel range of values, either 255 (int) or 1 (float)
    :type data_range: float or int
    :return: MSE, PSNR, SSIM values for each image
    :rtype: tuple
    """
    assert images.shape == ground_truths.shape, "Shapes of image batches must match"
    mse_values = torch.mean((images - ground_truths) ** 2, dim=(1, 2, 3)) 
    psnr_values = 10 * torch.log10(data_range ** 2 / mse_values)
    ssim_values = pt_ssim(images, ground_truths, data_range=data_range, size_average=False)
    #
    # Calc ncc and authentication success rate
    #
    ncc_values = torch.zeros(images.shape[0])
    asr_values = torch.zeros(images.shape[0])
    for i in range(images.shape[0]):
        ncc_values[i] = calc_ncc(images[i], ground_truths[i])
        if ncc_values[i] > 0.95:
            asr_values[i] = 1.0

    return {"mse": mse_values, "psnr": psnr_values, "ssim": ssim_values, "ncc": ncc_values, "asr": asr_values}


def calculate_mse_torch(images, ground_truths):
    """Calculate Mean Squared Error (MSE) for a batch of images using Pytorch tensors.

    :param images: batch of output images (PyTorch tensors)
    :type images: torch.Tensor
    :param ground_truths: batch of ground truth images (PyTorch tensors)
    :type ground_truths: torch.Tensor
    :return: MSE values for each image pair
    :rtype: float
    """    
    #
    # Ensure the images and ground truths have the same shape
    #
    assert images.shape == ground_truths.shape, "Shapes of image batches must match"
    #
    # Calculate MSE
    #
    mse_values = torch.mean((images - ground_truths) ** 2, dim=(1, 2, 3))
    return mse_values

def calculate_psnr_torch(images, ground_truths, data_range=1.0):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) for a batch of images using PyTorch tensors.

    :param images: batch of output images (PyTorch tensors)
    :type images: torch.Tensor
    :param ground_truths: batch of ground truth images (PyTorch tensors)
    :type ground_truths: torch.Tensor
    :param data_range: the data range of the pixel values (difference between the maximum and minimum possible values), defaults to 1.0
    :type data_range: float, optional
    :return: PSNR values for each image pair
    :rtype: float
    """    
    #
    # calc mse values
    #
    mse_values = calculate_mse_torch(images, ground_truths)
    #
    # calc psnr values
    #
    psnr_values = 10 * torch.log10(data_range ** 2 / mse_values)
    return psnr_values

def calculate_ssim_torch(images, ground_truths, data_range=1.0):
    """Calculate Structural Similarity Index Measure (SSIM) for a batch of images using PyTorch tensors.

    :param images: batch of output images (PyTorch tensors)
    :type images: torch.Tensor
    :param ground_truths: batch of ground truth images (PyTorch tensors)
    :type ground_truths: torch.Tensor
    :param data_range: the data range of the pixel values (difference between the maximum and minimum possible values), defaults to 1.0
    :type data_range: float, optional
    :return: SSIM values for each image pair
    :rtype: float
    """    
    #
    # calc psnr values
    #
    ssim_values = pt_ssim(images, ground_truths, data_range=data_range)
    return ssim_values


def norm_data(data):
    """Normalize data to have mean=0 and standard_deviation=1 for tensor images.

    :param data: PyTorch tensor of any shape
    :type data: torch.Tensor
    :return: normalized batch of images
    :rtype: torch.tensor(float)
    """    
    #
    # Get mean and std
    #
    mean_data = torch.mean(data)
    std_data = torch.std(data, unbiased=True)
    #
    # normlize data
    #
    if std_data == 0:
        return None  # Return None to indicate a uniform tensor
    normalized_data = (data - mean_data) / std_data
    return normalized_data

def calc_ncc(data0, data1):
    """Normalized correlation coefficient (NCC) between two tensor datasets.

    :param data0: PyTorch tensors of the same size
    :type data0: torch.Tensor
    :param data1: PyTorch tensors of the same size
    :type data1: torch.Tensor
    :return: NCC comparison between two images
    :rtype: float [-1, 1]
    """    
    #
    # ensure float
    #
    data0 = data0.float()
    data1 = data1.float()
    #
    # normalize data
    #
    norm_data0 = norm_data(data0)
    norm_data1 = norm_data(data1)
    #
    # Check if both tensors are uniform
    #
    if norm_data0 is None and norm_data1 is None:
        if torch.equal(data0, data1):
            return 1.0  # Perfect correlation for identical uniform images
        else:
            return -1.0  # Completely uncorrelated for different uniform images
    #
    # calculate ncc
    #
    ncc_value = torch.sum(norm_data0 * norm_data1) / (data0.numel() - 1)
    return ncc_value