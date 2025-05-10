import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

import curves


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def compute_metrics(output, target, num_classes):
    """Compute various metrics for model evaluation"""
    # Convert to numpy for sklearn metrics
    pred = output.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    probs = F.softmax(output, dim=1).cpu().numpy()
    
    # Compute accuracy
    accuracy = (pred == target).mean()
    
    # Compute F1 score (macro average)
    f1 = f1_score(target, pred, average='macro')
    
    # Compute ROC AUC (one-vs-rest)
    # TODO: fix ROC AUC, due to batching issues it does not work correctly.
    try:
        # Ensure we have all classes represented in the batch
        if len(np.unique(target)) == num_classes:
            roc_auc = roc_auc_score(target, probs, multi_class='ovr')
        else:
            # If not all classes are present, compute ROC AUC for each class separately
            roc_auc = 0.0
            for i in range(num_classes):
                try:
                    roc_auc += roc_auc_score((target == i).astype(int), probs[:, i])
                except ValueError:
                    continue
            roc_auc /= num_classes
    except ValueError:
        roc_auc = np.nan
    
    return {
        'accuracy': accuracy * 100,  # Convert to percentage
        'f1': f1 * 100,  # Convert to percentage
        'roc_auc': roc_auc * 100 if not np.isnan(roc_auc) else 0.0  # Convert to percentage
    }


def train(loader, model, optimizer, loss_tracker, regularizer=None, scaler=None):
    loss_tracker.reset()
    metrics = {
        'accuracy': 0.0,
        'f1': 0.0,
        'roc_auc': 0.0
    }
    num_processed = 0
    all_outputs = []
    all_targets = []

    for input, target in loader:
        input, target = input.cuda(), target.cuda()
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            output = model(input)
            
            # Store outputs and targets for metric computation
            all_outputs.append(output.detach())
            all_targets.append(target)
            
            # Compute main loss
            loss = loss_tracker.main_loss(output, target)
            
            # Add regularization if specified
            if regularizer is not None:
                loss = loss + regularizer(model)
        
        # Update loss tracker
        loss_tracker.update(output, target, input.size(0))
        
        optimizer.zero_grad()
        
        # Scale loss and backpropagate
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    batch_metrics = compute_metrics(all_outputs, all_targets, output.size(1))
    for k, v in batch_metrics.items():
        metrics[k] = v

    # Get final loss values
    losses = loss_tracker.get_losses()
    results = {
        'loss': losses['main'],
    }
    # Add metrics
    results.update(metrics)
    # Add auxiliary loss values
    results.update({k: v for k, v in losses.items() if k != 'main'})
    
    return results


def test(loader, model, loss_tracker, regularizer=None, scaler=None):
    loss_tracker.reset()
    metrics = {
        'accuracy': 0.0,
        'f1': 0.0,
        'roc_auc': 0.0
    }
    num_processed = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for input, target in loader:
            input, target = input.cuda(), target.cuda()
            
            # Use autocast for mixed precision
            with torch.cuda.amp.autocast():
                output = model(input)
                
                # Store outputs and targets for metric computation
                all_outputs.append(output.clone().detach())
                all_targets.append(target)
                
                # Compute main loss
                loss = loss_tracker.main_loss(output, target)
                
                # Add regularization if specified
                if regularizer is not None:
                    loss = loss + regularizer(model)
            
            # Update loss tracker
            loss_tracker.update(output, target, input.size(0))

    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    batch_metrics = compute_metrics(all_outputs, all_targets, output.size(1))
    for k, v in batch_metrics.items():
        metrics[k] = v

    # Get final loss values
    losses = loss_tracker.get_losses()
    results = {
        'loss': losses['main'],
    }
    # Add metrics
    results.update(metrics)
    # Add auxiliary loss values
    results.update({k: v for k, v in losses.items() if k != 'main'})
    
    return results


def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda(non_blocking=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))


def load_checkpoint(dir, epoch):
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    return torch.load(filepath)
