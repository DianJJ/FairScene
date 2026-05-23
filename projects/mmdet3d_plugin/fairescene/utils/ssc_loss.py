
import torch
import torch.nn as nn
import torch.nn.functional as F

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


# def geo_scal_loss(pred, ssc_target):
#
#     # Get softmax probabilities
#     pred = F.softmax(pred, dim=1)
#
#     # Compute empty and nonempty probabilities
#     empty_probs = pred[:, 0, :, :, :]
#     nonempty_probs = 1 - empty_probs
#
#     # Remove unknown voxels
#     mask = ssc_target != 255
#     nonempty_target = ssc_target != 0
#     nonempty_target = nonempty_target[mask].float()
#     nonempty_probs = nonempty_probs[mask]
#     empty_probs = empty_probs[mask]
#
#     intersection = (nonempty_target * nonempty_probs).sum()
#     precision = intersection / nonempty_probs.sum()
#     recall = intersection / nonempty_target.sum()
#     spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
#     return (
#         F.binary_cross_entropy(precision, torch.ones_like(precision))
#         + F.binary_cross_entropy(recall, torch.ones_like(recall))
#         + F.binary_cross_entropy(spec, torch.ones_like(spec))
#     )


def geo_scal_loss(pred, ssc_target,valid_mask):

    pred = F.softmax(pred, dim=-1)
    valid_mask = torch.tensor(valid_mask).unsqueeze(0)
    empty_probs = pred[:, :, :, :, 0]
    nonempty_probs = 1 - empty_probs

    nonempty_target = 1 - ssc_target[:, :, :, :, 0]

    nonempty_target = nonempty_target[valid_mask].float()
    nonempty_probs = nonempty_probs[valid_mask]
    empty_probs = empty_probs[valid_mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()

    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


# def geo_scal_loss(pred, ssc_target,valid_mask):
#
#     pred = F.softmax(pred, dim=-1)
#     valid_mask = torch.tensor(valid_mask).unsqueeze(0)
#     empty_probs = pred[:, :, :, :, 0]
#     nonempty_probs = 1 - empty_probs
#
#
#     nonempty_target = nonempty_target[valid_mask].float()
#     nonempty_probs = nonempty_probs[valid_mask]
#     empty_probs = empty_probs[valid_mask]
#
#     intersection = (nonempty_target * nonempty_probs).sum()
#     precision = intersection / nonempty_probs.sum()
#     recall = intersection / nonempty_target.sum()
#     spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
#
#     target_precision = torch.sum(nonempty_target * nonempty_target) / torch.sum(nonempty_target)
#     target_recall = target_precision
#     target_spec = torch.sum((1 - nonempty_target) * (1 - nonempty_target)) / torch.sum(1 - nonempty_target)
#
#     # return (
#     #     F.binary_cross_entropy(precision, torch.ones_like(precision))
#     #     + F.binary_cross_entropy(recall, torch.ones_like(recall))
#     #     + F.binary_cross_entropy(spec, torch.ones_like(spec))
#     # )
#     return (
#         F.binary_cross_entropy(precision, target_precision)
#         + F.binary_cross_entropy(recall, target_recall)
#         + F.binary_cross_entropy(spec, target_spec)
#     )

# def precision_loss(pred, ssc_target):
#
#     # Get softmax probabilities
#     pred = F.softmax(pred, dim=-1)
#
#     # Compute empty and nonempty probabilities
#     empty_probs = pred[:, 0, :, :, :]
#     nonempty_probs = 1 - empty_probs
#
#     # Remove unknown voxels
#     mask = ssc_target != 255
#     nonempty_target = ssc_target != 0
#     nonempty_target = nonempty_target[mask].float()
#     nonempty_probs = nonempty_probs[mask]
#     empty_probs = empty_probs[mask]
#
#     intersection = (nonempty_target * nonempty_probs).sum()
#     precision = intersection / nonempty_probs.sum()
#     return (
#         F.binary_cross_entropy(precision, torch.ones_like(precision))
#     )

# def sem_scal_loss(pred, ssc_target):
#     # Get softmax probabilities
#     pred = F.softmax(pred, dim=1)
#     loss = 0
#     count = 0
#     mask = ssc_target != 255
#     n_classes = pred.shape[1]
#     for i in range(0, n_classes):
#
#         # Get probability of class i
#         p = pred[:, i, :, :, :]
#
#         # Remove unknown voxels
#         target_ori = ssc_target
#         p = p[mask]
#         target = ssc_target[mask]
#
#         completion_target = torch.ones_like(target)
#         #pdb.set_trace()
#         completion_target[target != i] = 0
#         completion_target_ori = torch.ones_like(target_ori).float()
#         completion_target_ori[target_ori != i] = 0
#         if torch.sum(completion_target) > 0:
#             count += 1.0
#             nominator = torch.sum(p * completion_target)
#             loss_class = 0
#             if torch.sum(p) > 0:
#                 precision = nominator / (torch.sum(p))
#                 loss_precision = F.binary_cross_entropy(
#                     precision, torch.ones_like(precision)
#                 )
#                 loss_class += loss_precision
#             if torch.sum(completion_target) > 0:
#                 recall = nominator / (torch.sum(completion_target))
#                 loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
#
#
#                 loss_class += loss_recall
#             if torch.sum(1 - completion_target) > 0:
#                 specificity = torch.sum((1 - p) * (1 - completion_target)) / (
#                     torch.sum(1 - completion_target)
#                 )
#                 loss_specificity = F.binary_cross_entropy(
#                     specificity, torch.ones_like(specificity)
#                 )
#                 loss_class += loss_specificity
#             loss += loss_class
#     return loss / count


def sem_scal_loss(pred, ssc_target,valid_mask):

    pred = F.softmax(pred, dim=-1)
    loss = 0
    count = 0
    valid_mask = torch.tensor(valid_mask).unsqueeze(0)
    # mask = ssc_target.sum(dim=1) != 0  # [batch_size, D, H, W]

    n_classes = pred.shape[-1]

    for i in range(0, n_classes):
        p = pred[:, :, :, :, i]
        p = p[valid_mask]
        target = ssc_target
        completion_target = target[:, :, :, :, i][valid_mask]

        # p = p[mask]
        # completion_target = completion_target[mask]

        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0

            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                target_precision = torch.sum(completion_target * completion_target) / torch.sum(completion_target)
                #loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
                loss_precision = F.binary_cross_entropy(precision, torch.ones_like(target_precision))
                loss_class += loss_precision

            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                target_recall = torch.sum(completion_target * completion_target) / torch.sum(completion_target)
                #loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_recall = F.binary_cross_entropy(recall, target_recall)
                loss_class += loss_recall

            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / torch.sum(1 - completion_target)
                loss_specificity = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
                loss_class += loss_specificity

            loss += loss_class

    return loss / count

# def sem_scal_loss(pred, ssc_target,valid_mask):
#
#     pred = F.softmax(pred, dim=-1)
#     loss = 0
#     count = 0
#     valid_mask = torch.tensor(valid_mask).unsqueeze(0)
#     # mask = ssc_target.sum(dim=1) != 0  # [batch_size, D, H, W]
#
#     n_classes = pred.shape[-1]
#
#     for i in range(0, n_classes):
#         p = pred[:, :, :, :, i]
#         p = p[valid_mask]
#         target = ssc_target
#         completion_target = target[:, :, :, :, i][valid_mask]
#
#         # p = p[mask]
#         # completion_target = completion_target[mask]
#
#         if torch.sum(completion_target) > 0:
#             count += 1.0
#             nominator = torch.sum(p * completion_target)
#             loss_class = 0
#
#             if torch.sum(p) > 0:
#                 precision = nominator / (torch.sum(p))
#                 target_precision = torch.sum(completion_target * completion_target) / torch.sum(completion_target)
#                 #loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
#                 loss_precision = F.binary_cross_entropy(precision, target_precision)
#                 loss_class += loss_precision
#
#             if torch.sum(completion_target) > 0:
#                 recall = nominator / (torch.sum(completion_target))
#                 target_recall = torch.sum(completion_target * completion_target) / torch.sum(completion_target)
#                 #loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
#                 loss_recall = F.binary_cross_entropy(recall, target_recall)
#                 loss_class += loss_recall
#
#             if torch.sum(1 - completion_target) > 0:
#                 specificity = torch.sum((1 - p) * (1 - completion_target)) / torch.sum(1 - completion_target)
#                 target_specificity = torch.sum((1 - completion_target) * (1 - completion_target)) / torch.sum(
#                     1 - completion_target)
#                 #loss_specificity = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
#                 loss_specificity = F.binary_cross_entropy(specificity, target_specificity)
#                 loss_class += loss_specificity
#
#             loss += loss_class
#
#     return loss / count

# def CE_ssc_loss(pred, target, class_weights):
#
#     criterion = nn.CrossEntropyLoss(
#         weight=class_weights, ignore_index=255, reduction="none"
#     )
#     loss = criterion(pred, target.long())
#     loss_valid = loss[target!=255]
#     loss_valid_mean = torch.mean(loss_valid)
#     return loss_valid_mean


def CE_ssc_loss(pred, target, class_weights, valid_mask):
    """Cross-entropy loss for one-hot encoded targets.

    Args:
        pred: [N, H, W, D, C] logits (N: batch, H/W/D: voxel grid, C: classes).
        target: one-hot encoded targets [N, H, W, D, C].
        class_weights: per-class weights [C].
        valid_mask: boolean mask selecting voxels to include in the loss.
    """

    # L = - sum_c y_c * log p_c
    if class_weights is not None:
        class_weights = class_weights.view(1, 1, 1, 1, -1)
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(target * log_prob * class_weights, dim=-1).squeeze()

    else:
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(target * log_prob, dim=-1).squeeze()


    loss_valid = loss[valid_mask]
    loss_valid_mean = torch.mean(loss_valid)

    return loss_valid_mean

def BCE_ssc_loss(pred, target, class_weights, alpha):

    class_weights[0] = 1-alpha    # empty                 
    class_weights[1] = alpha    # occupied                      

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)

    return loss_valid_mean





# def CE_ssc_loss(pred, target, class_weights):
#
#     criterion = nn.CrossEntropyLoss(
#         weight=class_weights, ignore_index=255, reduction="none"
#     )
#     loss = criterion(pred, target.long())
#     loss_valid = loss[target!=255]
#     loss_valid_mean = torch.mean(loss_valid)
#     return loss_valid_mean

