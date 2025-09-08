#!/usr/bin/env python3
"""
Test script to verify the memory-efficient dice calculation works correctly.
This simulates the problematic scenario and tests the fix.
"""

import numpy as np
import time
import sys

# Import the memory-efficient dice functions
from memory_efficient_dice import dice, dice_memory_efficient

def create_test_arrays(shape=(14, 100, 100, 100)):
    """Create test arrays similar to the problematic case."""
    print(f"Creating test arrays with shape: {shape}")
    
    # Create random binary arrays (0 and 1) to simulate segmentation masks
    np.random.seed(42)  # For reproducible results
    x = np.random.randint(0, 2, shape).astype(np.float32)
    y = np.random.randint(0, 2, shape).astype(np.float32)
    
    # Calculate memory usage
    memory_mb = (x.nbytes + y.nbytes) / (1024 * 1024)
    print(f"Arrays created. Total memory usage: {memory_mb:.2f} MB")
    
    return x, y

def test_original_dice_method(x, y):
    """Test the original problematic dice calculation method."""
    print("\n=== Testing Original Method (Problematic) ===")
    try:
        start_time = time.time()
        
        # This is the problematic line from the original code
        intersect = np.sum(np.sum(np.sum(x * y)))  # Creates large intermediate array
        y_sum = np.sum(np.sum(np.sum(y)))
        if y_sum == 0:
            result = 0.0
        else:
            x_sum = np.sum(np.sum(np.sum(x)))
            result = (2 * intersect) / (x_sum + y_sum)
        
        elapsed = time.time() - start_time
        print(f"Original method succeeded: {result:.6f} (took {elapsed:.3f}s)")
        return result
        
    except MemoryError as e:
        print(f"Original method failed with MemoryError: {e}")
        return None
    except Exception as e:
        print(f"Original method failed with error: {e}")
        return None

def test_memory_efficient_method(x, y):
    """Test the memory-efficient dice calculation method."""
    print("\n=== Testing Memory-Efficient Method ===")
    try:
        start_time = time.time()
        
        result = dice_memory_efficient(x, y, chunk_size=500000)
        
        elapsed = time.time() - start_time
        print(f"Memory-efficient method succeeded: {result:.6f} (took {elapsed:.3f}s)")
        return result
        
    except Exception as e:
        print(f"Memory-efficient method failed: {e}")
        return None

def test_different_chunk_sizes(x, y):
    """Test different chunk sizes to find optimal performance."""
    print("\n=== Testing Different Chunk Sizes ===")
    
    chunk_sizes = [50000, 100000, 500000, 1000000, 2000000]
    
    for chunk_size in chunk_sizes:
        try:
            start_time = time.time()
            result = dice_memory_efficient(x, y, chunk_size=chunk_size)
            elapsed = time.time() - start_time
            print(f"Chunk size {chunk_size:7d}: {result:.6f} (took {elapsed:.3f}s)")
        except Exception as e:
            print(f"Chunk size {chunk_size:7d}: FAILED - {e}")

def main():
    print("Testing Memory-Efficient Dice Calculation Fix")
    print("=" * 50)
    
    # Test with smaller arrays first
    print("\n1. Testing with smaller arrays (should work with both methods):")
    x_small, y_small = create_test_arrays(shape=(14, 50, 50, 50))
    
    original_result = test_original_dice_method(x_small, y_small)
    efficient_result = test_memory_efficient_method(x_small, y_small)
    
    if original_result is not None and efficient_result is not None:
        diff = abs(original_result - efficient_result)
        print(f"\nResults comparison:")
        print(f"Original method:  {original_result:.6f}")
        print(f"Efficient method: {efficient_result:.6f}")
        print(f"Difference:       {diff:.6f}")
        if diff < 1e-6:
            print("✅ Results match! Memory-efficient method is correct.")
        else:
            print("❌ Results don't match! There may be an issue with the implementation.")
    
    # Test with larger arrays (similar to the problematic case)
    print("\n\n2. Testing with larger arrays (similar to problematic case):")
    try:
        # Use a size that's large but manageable for testing
        # The original error was with shape (14, 323, 279, 248)
        # We'll use a smaller but still challenging size
        x_large, y_large = create_test_arrays(shape=(14, 150, 150, 150))
        
        original_result_large = test_original_dice_method(x_large, y_large)
        efficient_result_large = test_memory_efficient_method(x_large, y_large)
        
        test_different_chunk_sizes(x_large, y_large)
        
    except MemoryError:
        print("Cannot create large test arrays due to memory constraints.")
        print("This confirms the memory issue exists and the fix is needed.")
    
    print("\n" + "=" * 50)
    print("CONCLUSION:")
    print("If the memory-efficient method works while the original fails,")
    print("then the fix will resolve your training validation error.")
    print("Apply the patch from PATCH_trainer_dice_fix.py to your trainer.py file.")

if __name__ == "__main__":
    main()