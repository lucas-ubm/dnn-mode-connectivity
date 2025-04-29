import torch
import torch.nn as nn

class LossTracker:
    def __init__(self, main_loss, auxiliary_losses=None):
        """
        Initialize the loss tracker with a main loss function and optional auxiliary losses.
        
        Args:
            main_loss (nn.Module): The main loss function (e.g., CrossEntropyLoss)
            auxiliary_losses (dict): Dictionary of auxiliary loss functions with their names as keys
        """
        self.main_loss = main_loss
        self.auxiliary_losses = auxiliary_losses or {}
        
        # Initialize running sums
        self.reset()
    
    def reset(self):
        """Reset all running sums to zero."""
        self.main_loss_sum = 0.0
        self.aux_loss_sums = {name: 0.0 for name in self.auxiliary_losses}
        self.num_samples = 0
    
    def update(self, main_loss, aux_losses, batch_size):
        """
        Update running sums with new loss values.
        
        Args:
            main_loss (torch.Tensor): Value of the main loss
            aux_losses (dict): Dictionary of auxiliary loss values
            batch_size (int): Number of samples in the current batch
        """
        # Update main loss
        if isinstance(main_loss, torch.Tensor):
            self.main_loss_sum += main_loss.item() * batch_size
        else:
            self.main_loss_sum += main_loss * batch_size
        
        # Update auxiliary losses
        for name, loss in aux_losses.items():
            if isinstance(loss, torch.Tensor):
                self.aux_loss_sums[name] += loss.item() * batch_size
            else:
                self.aux_loss_sums[name] += loss * batch_size
        
        self.num_samples += batch_size
    
    def get_main_loss(self):
        """Get the average main loss."""
        return self.main_loss_sum / self.num_samples if self.num_samples > 0 else 0.0
    
    def get_auxiliary_losses(self):
        """Get the average auxiliary losses."""
        return {
            name: loss_sum / self.num_samples if self.num_samples > 0 else 0.0
            for name, loss_sum in self.aux_loss_sums.items()
        }
    
    def get_all_losses(self):
        """Get both main and auxiliary losses."""
        losses = {'main': self.get_main_loss()}
        losses.update(self.get_auxiliary_losses())
        return losses 