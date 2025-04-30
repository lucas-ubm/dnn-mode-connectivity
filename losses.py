import torch
import torch.nn.functional as F

class BaseLoss:
    """Base class for all loss functions"""
    def __init__(self, reduction='mean', name=None):
        self.reduction = reduction
        self.name = name or self.__class__.__name__

    def __call__(self, output, target):
        raise NotImplementedError

class CrossEntropyLoss(BaseLoss):
    """Standard cross entropy loss"""
    def __call__(self, output, target):
        return F.cross_entropy(output, target, reduction=self.reduction)

class FocalLoss(BaseLoss):
    """Focal Loss for handling class imbalance
    
    Parameters:
        alpha (float): Weight for the rare class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Reduction method ('none', 'mean', 'sum')
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', name=None):
        super().__init__(reduction, name)
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, output, target):
        ce_loss = F.cross_entropy(output, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedLoss(BaseLoss):
    """Wrapper to add class weights to any loss function"""
    def __init__(self, base_loss, weights, name=None):
        super().__init__(name=name)
        self.base_loss = base_loss
        self.weights = weights

    def __call__(self, output, target):
        loss = self.base_loss(output, target)
        if self.reduction == 'none':
            return loss * self.weights[target]
        return loss

class LossTracker:
    """Tracks multiple losses during training"""
    def __init__(self, main_loss, auxiliary_losses=None):
        self.main_loss = main_loss
        self.auxiliary_losses = auxiliary_losses or {}
        self.reset()

    def reset(self):
        """Reset all loss accumulators"""
        self.main_loss_sum = 0.0
        self.aux_loss_sums = {name: 0.0 for name in self.auxiliary_losses}
        self.num_samples = 0

    def update(self, output, target, batch_size):
        """Update loss accumulators with a new batch"""
        # Compute main loss
        if isinstance(target, dict):
            target_tensor = target['target']
        else:
            target_tensor = target
            
        main_loss = self.main_loss(output, target_tensor)
        if isinstance(main_loss, torch.Tensor):
            self.main_loss_sum += main_loss.item() * batch_size
        else:
            self.main_loss_sum += main_loss * batch_size

        # Compute auxiliary losses
        for name, loss_fn in self.auxiliary_losses.items():
            aux_loss = loss_fn(output, target_tensor)
            if isinstance(aux_loss, torch.Tensor):
                self.aux_loss_sums[name] += aux_loss.item() * batch_size
            else:
                self.aux_loss_sums[name] += aux_loss * batch_size

        self.num_samples += batch_size

    def get_losses(self):
        """Get all current loss values"""
        losses = {
            'main': self.main_loss_sum / self.num_samples
        }
        losses.update({
            name: loss_sum / self.num_samples 
            for name, loss_sum in self.aux_loss_sums.items()
        })
        return losses

def get_loss(loss_name, **kwargs):
    """Factory function to create loss instances
    
    Parameters:
        loss_name (str): Name of the loss function ('ce', 'focal', 'weighted')
        **kwargs: Additional parameters for the loss function
    
    Returns:
        BaseLoss: Instance of the requested loss function
    """
    loss_dict = {
        'ce': CrossEntropyLoss,
        'focal': FocalLoss,
    }
    
    if loss_name not in loss_dict:
        raise ValueError(f"Loss {loss_name} not found. Available losses: {list(loss_dict.keys())}")
    
    return loss_dict[loss_name](**kwargs) 