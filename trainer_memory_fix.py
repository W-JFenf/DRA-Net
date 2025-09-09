# Memory-efficient fix for trainer.py
# Replace the problematic dice function with this memory-efficient version

import numpy as np

def dice_memory_efficient(x, y, chunk_size=500000):
    """
    Memory-efficient dice coefficient calculation.
    Processes large arrays in chunks to avoid memory allocation errors.
    
    This replaces the original dice function that caused:
    numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.17 GiB
    """
    # Ensure arrays are the same shape
    assert x.shape == y.shape, f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
    
    # Flatten arrays for efficient processing
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
        
        # Calculate intersection for this chunk (avoids creating large intermediate arrays)
        intersect_chunk = np.sum(x_chunk * y_chunk)
        total_intersect += intersect_chunk
        
        # Calculate sums for this chunk
        total_x_sum += np.sum(x_chunk)
        total_y_sum += np.sum(y_chunk)
    
    # Calculate dice coefficient with smoothing
    dice_coeff = (2.0 * total_intersect + 1e-5) / (total_x_sum + total_y_sum + 1e-5)
    return dice_coeff


# Alternative even more memory-efficient version using iterative calculation
def dice_ultra_efficient(x, y):
    """
    Ultra memory-efficient dice calculation that processes element by element.
    Use this if the chunk-based approach still causes memory issues.
    """
    # Flatten arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    intersect = 0
    x_sum = 0
    y_sum = 0
    
    # Process element by element to minimize memory usage
    for i in range(len(x_flat)):
        x_val = x_flat[i]
        y_val = y_flat[i]
        intersect += x_val * y_val
        x_sum += x_val
        y_sum += y_val
    
    return (2.0 * intersect + 1e-5) / (x_sum + y_sum + 1e-5)


# Drop-in replacement for the original dice function
def dice(x, y):
    """
    REPLACE THE ORIGINAL DICE FUNCTION IN TRAINER.PY WITH THIS VERSION
    
    Original problematic code:
    def dice(x, y):
        intersect = np.sum(np.sum(np.sum(x * y)))  # <-- This line causes memory error
        y_sum = np.sum(np.sum(np.sum(y)))
        if y_sum == 0:
            return 0.0
        x_sum = np.sum(np.sum(np.sum(x)))
        return (2 * intersect) / (x_sum + y_sum)
    """
    try:
        # Try the chunk-based approach first
        return dice_memory_efficient(x, y, chunk_size=500000)
    except MemoryError:
        # Fallback to ultra-efficient approach if still not enough memory
        print("Using ultra-efficient dice calculation due to memory constraints")
        return dice_ultra_efficient(x, y)


# Instructions for implementation:
"""
TO FIX THE MEMORY ERROR IN YOUR TRAINER.PY:

1. Replace the existing dice function (around line 26 in trainer.py) with the dice() function above

2. The original problematic function:
   def dice(x, y):
       intersect = np.sum(np.sum(np.sum(x * y)))  # This creates a 1.17GB array!
       y_sum = np.sum(np.sum(np.sum(y)))
       if y_sum == 0:
           return 0.0
       x_sum = np.sum(np.sum(np.sum(x)))
       return (2 * intersect) / (x_sum + y_sum)

3. Should be replaced with:
   def dice(x, y):
       try:
           return dice_memory_efficient(x, y, chunk_size=500000)
       except MemoryError:
           print("Using ultra-efficient dice calculation due to memory constraints")
           return dice_ultra_efficient(x, y)

4. Add the helper functions (dice_memory_efficient, dice_ultra_efficient) at the top of your trainer.py file

EXPLANATION OF THE FIX:
- The original code creates x * y which is a huge array (14, 323, 279, 248) = 1.17GB
- The new code processes data in chunks, never creating the full multiplication array at once
- Chunk size of 500,000 elements keeps memory usage manageable
- Fallback to element-by-element processing if chunks are still too large
"""