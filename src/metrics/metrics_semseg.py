import segmentation_models_pytorch as smp
import torch


def calculate_semseg_metrics(pred_masks, target):
    """Calculate semantic segmentation metrics, including accuracy, iou, f1-score, and recall metrics. 

    :param pred_masks: predicted semantic segmentation masks (size of the imagex)
    :type pred_masks: torch.tensor
    :param target: true semantic segmentation mask
    :type target: torch.tensor
    :return: accuracy, iou, f1-score, recall metrics comparing the predicted mask to the acutal mask
    :rtype: dict
    """    
    # make sure everythin is in the right format
    if len(target.shape) == 3:
        target = target.unsqueeze(1)
        
    pred_masks = torch.argmax(pred_masks, dim=1)

    if len(pred_masks.shape) == 3:
        pred_masks = pred_masks.unsqueeze(1)

    # first compute statistics for true positives, false positives, false negative and
    # true negative "pixels"
    tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, target, mode='binary', threshold=0.5)

    # then compute metrics with required reduction (see metric docs)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

    return {"iou_score": iou_score, "f1_score": f1_score, "acc": accuracy, "recall": recall}