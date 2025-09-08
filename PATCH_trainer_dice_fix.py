"""
MEMORY FIX PATCH FOR TRAINER.PY

This patch fixes the numpy._ArrayMemoryError in the dice calculation.

PROBLEM:
Line 26 in trainer.py: intersect = np.sum(np.sum(np.sum(x * y)))
Creates a 1.17GB array (14, 323, 279, 248) causing memory allocation failure.

SOLUTION:
Replace the dice function with a memory-efficient chunk-based implementation.
"""

import numpy as np

# ============================================================================
# STEP 1: Add these helper functions to the top of your trainer.py file
# ============================================================================

def dice_memory_efficient(x, y, chunk_size=500000):
    """Memory-efficient dice coefficient calculation using chunks."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    total_intersect = 0
    total_x_sum = 0
    total_y_sum = 0
    
    # Process in chunks to avoid creating large intermediate arrays
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
    
    # Handle division by zero
    if total_y_sum == 0:
        return 0.0
    
    # Calculate dice coefficient
    dice_coeff = (2.0 * total_intersect) / (total_x_sum + total_y_sum)
    return dice_coeff

# ============================================================================
# STEP 2: Replace your existing dice function with this one
# ============================================================================

def dice(x, y):
    """
    REPLACE YOUR EXISTING DICE FUNCTION WITH THIS VERSION
    
    OLD PROBLEMATIC VERSION:
    def dice(x, y):
        intersect = np.sum(np.sum(np.sum(x * y)))  # <-- MEMORY ERROR HERE
        y_sum = np.sum(np.sum(np.sum(y)))
        if y_sum == 0:
            return 0.0
        x_sum = np.sum(np.sum(np.sum(x)))
        return (2 * intersect) / (x_sum + y_sum)
    """
    return dice_memory_efficient(x, y, chunk_size=500000)


# ============================================================================
# IMPLEMENTATION INSTRUCTIONS:
# ============================================================================

"""
TO APPLY THIS FIX:

1. Open your trainer.py file

2. Find the existing dice function (around line 26):
   def dice(x, y):
       intersect = np.sum(np.sum(np.sum(x * y)))
       y_sum = np.sum(np.sum(np.sum(y)))
       if y_sum == 0:
           return 0.0
       x_sum = np.sum(np.sum(np.sum(x)))
       return (2 * intersect) / (x_sum + y_sum)

3. Replace it with:
   def dice(x, y):
       return dice_memory_efficient(x, y, chunk_size=500000)

4. Add the dice_memory_efficient function definition before the dice function

5. Save and run your training again

OPTIONAL OPTIMIZATIONS:
- If you still get memory errors, reduce chunk_size to 100000 or 50000
- If you have more memory available, increase chunk_size to 1000000 for faster processing
"""

# ============================================================================
# ALTERNATIVE: Complete replacement for the validation section
# ============================================================================

def dice_validation_memory_safe(val_output_convert, val_labels_convert):
    """
    Memory-safe version of the validation dice calculation.
    Use this to replace the problematic line in val_epoch function.
    
    REPLACE THIS LINE:
    dice_score = dice(val_output_convert[0].cpu().numpy(), val_labels_convert[0].cpu().numpy())
    
    WITH:
    dice_score = dice_validation_memory_safe(val_output_convert, val_labels_convert)
    """
    pred_np = val_output_convert[0].cpu().numpy()
    target_np = val_labels_convert[0].cpu().numpy()
    
    return dice_memory_efficient(pred_np, target_np, chunk_size=500000)