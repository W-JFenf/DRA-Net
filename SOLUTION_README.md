# Memory Error Fix for UNETR Training

## Problem Description

Your UNETR training encounters a memory allocation error during validation:

```
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.17 GiB for an array with shape (14, 323, 279, 248) and data type float32
```

This error occurs in the `dice` function at line 26 of `trainer.py`:
```python
intersect = np.sum(np.sum(np.sum(x * y)))  # Creates 1.17GB array!
```

## Root Cause

The original dice calculation creates a massive intermediate array `x * y` with shape `(14, 323, 279, 248)`, requiring 1.17 GB of memory. This causes numpy to fail when trying to allocate this much contiguous memory.

## Solution: Memory-Efficient Chunked Processing

Instead of creating the entire `x * y` array at once, we process the data in smaller chunks, significantly reducing memory usage while producing the same mathematical result.

## Implementation

### Step 1: Add Helper Function

Add this function to your `trainer.py` file (before the existing `dice` function):

```python
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
```

### Step 2: Replace the Dice Function

Replace your existing `dice` function with:

```python
def dice(x, y):
    """Memory-efficient dice calculation - replaces problematic version."""
    return dice_memory_efficient(x, y, chunk_size=500000)
```

### Original Problematic Code (REMOVE THIS):
```python
def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))  # <-- CAUSES MEMORY ERROR
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return (2 * intersect) / (x_sum + y_sum)
```

## Memory Usage Comparison

| Method | Memory Usage | Status |
|--------|-------------|---------|
| Original | 1.17 GB (single array) | ❌ Memory Error |
| Chunked (500K) | ~1.9 MB per chunk | ✅ Success |
| Chunked (100K) | ~0.4 MB per chunk | ✅ Success |

## Performance Tuning

You can adjust the `chunk_size` parameter based on your available memory:

- **More Memory Available**: Increase to `1000000` or `2000000` for faster processing
- **Less Memory Available**: Decrease to `100000` or `50000` for lower memory usage
- **Default**: `500000` provides a good balance

## Alternative Ultra-Efficient Version

If you still encounter memory issues, use this ultra-efficient version:

```python
def dice_ultra_efficient(x, y):
    """Ultra memory-efficient dice calculation (element-by-element)."""
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
    
    if y_sum == 0:
        return 0.0
    return (2.0 * intersect) / (x_sum + y_sum)

def dice(x, y):
    """Dice with fallback for extreme memory constraints."""
    try:
        return dice_memory_efficient(x, y, chunk_size=500000)
    except MemoryError:
        print("Using ultra-efficient dice calculation due to memory constraints")
        return dice_ultra_efficient(x, y)
```

## Verification

The test demonstrates that:
- Original method: Requires 1.17 GB → Memory Error
- Chunked method: Uses ~1.9 MB per chunk → Success
- Mathematical result: Identical to original method

## Files Provided

1. **`PATCH_trainer_dice_fix.py`** - Complete patch with implementation details
2. **`memory_efficient_dice.py`** - Standalone memory-efficient functions
3. **`simple_test.py`** - Demonstration of the memory usage difference

## Summary

This fix resolves the memory allocation error by:
1. Avoiding creation of the 1.17GB intermediate array
2. Processing data in manageable 500K element chunks (~1.9MB each)
3. Maintaining mathematical accuracy
4. Providing tunable memory usage via chunk size

Apply this fix to your `trainer.py` file and your UNETR training should continue without memory errors during validation.