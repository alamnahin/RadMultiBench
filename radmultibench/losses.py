import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implements Label Smoothing Cross Entropy Loss.
    [cite_start][cite: 188-189]
    """

    def __init__(self, epsilon=0.1, num_classes=5):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Args:
        alpha: Class weights (tensor of shape [num_classes]) or None
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions (logits) of shape [batch_size, num_classes]
            targets: ground truth labels of shape [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingFocalLoss(nn.Module):
    """
    Combines Label Smoothing and Focal Loss for better handling of 
    class imbalance with regularization.
    """

    def __init__(self, epsilon=0.1, num_classes=5, alpha=None, gamma=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: predictions (logits) of shape [batch_size, num_classes]
            target: ground truth labels of shape [batch_size]
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
        
        # Compute cross entropy with smoothed labels
        ce_loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            ce_loss = alpha_t * ce_loss
        
        # Compute focal weight
        probs = torch.exp(log_probs)
        pt = torch.sum(true_dist * probs, dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        
        # Final loss
        focal_loss = focal_weight * ce_loss
        
        return torch.mean(focal_loss)


def compute_class_weights(class_counts, device='cuda'):
    """
    Compute inverse frequency class weights for handling imbalance.
    
    Args:
        class_counts: list or array of counts for each class
        device: torch device
    
    Returns:
        torch.Tensor of class weights
    """
    total = sum(class_counts)
    num_classes = len(class_counts)
    weights = torch.tensor(
        [total / (num_classes * c) for c in class_counts],
        dtype=torch.float32
    )
    return weights.to(device)


def build_criterion(cfg, class_counts=None):
    """
    Factory function for building criterion.
    
    Args:
        cfg: configuration object
        class_counts: list/array of class counts for computing weights (optional)
    
    Returns:
        Loss function
    """
    task = cfg.TASK

    if task == "classification":
        # Compute class weights if provided
        class_weights = None
        if class_counts is not None:
            class_weights = compute_class_weights(class_counts, device=cfg.DEVICE)
            print(f"[INFO] Using class weights: {class_weights}")
        
        # Choose loss type based on whether we have class imbalance
        if class_weights is not None:
            # Use Focal Loss with class weights for imbalanced datasets
            print("[INFO] Using Focal Loss with class weights for classification")
            return FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
        else:
            # Use Label Smoothing for balanced datasets
            print("[INFO] Using Label Smoothing Cross Entropy for classification")
            return LabelSmoothingCrossEntropy(
                epsilon=cfg.SOLVER.LABEL_SMOOTHING, 
                num_classes=cfg.MODEL.NUM_CLASSES
            )
    
    elif task == "generation":
        # For ResNet-LSTM
        if cfg.MODEL.NAME == "resnet_lstm_generation":
            print("[INFO] Using CrossEntropyLoss for ResNet-LSTM generation")
            return nn.CrossEntropyLoss(ignore_index=cfg.DATA.PAD_IDX)
        # For CLIP-GPT and BioViL-T, loss is computed inside the model
        print("[INFO] Loss computed inside model for generation")
        return None
    
    else:
        raise ValueError(f"Unknown task: {task}")
