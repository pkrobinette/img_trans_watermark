import segmentation_models_pytorch as smp
import torch


def calculate_classification_metrics(preds, target):
    """Calculate accuracy, escape, and overkill metrics.

    :param preds: a batch of images from a model
    :type preds: torch.tensor
    :param target: the true labels of the model (reconstruction == input image)
    :type target: torch.tensor
    :return: accuracy, escape, overkill of predicted to true label
    :rtype: dict
    """    
    preds = torch.argmax(preds, dim=1)
    correct = [1 if y == p else 0 for y, p in zip(target, preds)]

    accuracy = sum(correct) / len(correct)
    #
    # Number of watermarks to not be classified as a watermark
    # e.g., they escaped noticed. False Negatives.
    #
    escape = sum(1 for t, p in zip(target, preds) if t == 1 and p == 0) / len(correct)
    #
    # Number of normal images to be classified as a watermark.
    # e.g., overkill. False Positives.
    #
    overkill = sum(1 for t, p in zip(target, preds) if t == 0 and p == 1) / len(correct)

    return {"accuracy": accuracy, "escape": escape, "overkill": overkill}