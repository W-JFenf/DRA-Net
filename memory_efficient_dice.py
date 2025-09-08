# Memory-efficient dice calculation functions for UNETR training
# This addresses the numpy._ArrayMemoryError by processing data in chunks

import numpy as np
import torch


def dice_memory_efficient(x, y, chunk_size=1000000):
    """
    Memory-efficient dice coefficient calculation.
    Processes large arrays in chunks to avoid memory allocation errors.
    
    Args:
        x: prediction array (numpy array)
        y: ground truth array (numpy array)
        chunk_size: size of chunks to process at a time
    
    Returns:
        dice coefficient (float)
    """
    # Ensure arrays are flattened for chunk processing
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    total_intersect = 0
    total_x_sum = 0
    total_y_sum = 0
    
    # Process in chunks to avoid memory issues
    for i in range(0, len(x_flat), chunk_size):
        end_idx = min(i + chunk_size, len(x_flat))
        x_chunk = x_flat[i:end_idx]
        y_chunk = y_flat[i:end_idx]
        
        # Calculate intersection for this chunk
        intersect_chunk = np.sum(x_chunk * y_chunk)
        total_intersect += intersect_chunk
        
        # Calculate sums for this chunk
        total_x_sum += np.sum(x_chunk)
        total_y_sum += np.sum(y_chunk)
    
    # Calculate dice coefficient
    dice_coeff = (2.0 * total_intersect + 1e-5) / (total_x_sum + total_y_sum + 1e-5)
    return dice_coeff


def dice_per_class_memory_efficient(pred, target, num_classes=14, chunk_size=1000000):
    """
    Memory-efficient per-class dice coefficient calculation.
    
    Args:
        pred: prediction tensor/array with shape (C, H, W, D) where C is number of classes
        target: ground truth tensor/array with same shape as pred
        num_classes: number of classes
        chunk_size: size of chunks to process at a time
    
    Returns:
        list of dice coefficients for each class
    """
    dice_scores = []
    
    for class_idx in range(num_classes):
        if isinstance(pred, torch.Tensor):
            pred_class = pred[class_idx].cpu().numpy()
        else:
            pred_class = pred[class_idx]
            
        if isinstance(target, torch.Tensor):
            target_class = target[class_idx].cpu().numpy()
        else:
            target_class = target[class_idx]
        
        dice_score = dice_memory_efficient(pred_class, target_class, chunk_size)
        dice_scores.append(dice_score)
    
    return dice_scores


def dice_optimized_torch(pred, target, smooth=1e-5):
    """
    Optimized torch-based dice calculation that avoids creating large intermediate arrays.
    
    Args:
        pred: prediction tensor
        target: ground truth tensor
        smooth: smoothing factor
    
    Returns:
        dice coefficient
    """
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection using torch operations (more memory efficient)
    intersection = torch.sum(pred_flat * target_flat)
    
    # Calculate dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice.item()


# Drop-in replacement for the problematic dice function in trainer.py
def dice(x, y):
    """
    Memory-efficient replacement for the original dice function.
    This function processes data in chunks to avoid memory allocation errors.
    
    Args:
        x: prediction array (numpy array)
        y: ground truth array (numpy array)
    
    Returns:
        dice coefficient (float)
    """
    return dice_memory_efficient(x, y, chunk_size=500000)  # Smaller chunk size for safety


# Alternative implementation using boolean operations (more memory efficient)
def dice_boolean(x, y):
    """
    Boolean-based dice calculation that's more memory efficient.
    
    Args:
        x: prediction array (numpy array)
        y: ground truth array (numpy array)
    
    Returns:
        dice coefficient (float)
    """
    # Convert to boolean to save memory
    x_bool = x.astype(bool)
    y_bool = y.astype(bool)
    
    # Calculate intersection using boolean operations
    intersection = np.logical_and(x_bool, y_bool).sum()
    
    # Calculate dice coefficient
    dice_coeff = (2.0 * intersection + 1e-5) / (x_bool.sum() + y_bool.sum() + 1e-5)
    return dice_coeff