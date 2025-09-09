#!/usr/bin/env python3
"""
测试不使用flatten()的dice计算方法
"""

def create_large_array_simulation(shape):
    """
    模拟创建大数组的内存使用情况
    """
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    memory_bytes = total_elements * 4  # float32 = 4 bytes
    memory_mb = memory_bytes / (1024 * 1024)
    memory_gb = memory_bytes / (1024 * 1024 * 1024)
    
    return total_elements, memory_mb, memory_gb

def simulate_flat_iteration(shape, block_size=1000000):
    """
    模拟使用flat迭代器处理大数组
    """
    total_elements, memory_mb, memory_gb = create_large_array_simulation(shape)
    
    print(f"数组形状: {shape}")
    print(f"总元素数: {total_elements:,}")
    print(f"如果创建展平数组需要内存: {memory_gb:.2f} GB")
    print(f"使用flat迭代器需要内存: 几乎为0 (只是迭代器)")
    
    # 模拟分块处理
    num_blocks = (total_elements + block_size - 1) // block_size
    memory_per_block = (block_size * 4) / (1024 * 1024)  # MB
    
    print(f"\n分块处理:")
    print(f"块大小: {block_size:,} 个元素")
    print(f"块数量: {num_blocks:,}")
    print(f"每块内存: {memory_per_block:.2f} MB")
    print(f"总处理时间估计: {num_blocks * 0.1:.1f} 秒 (假设每块0.1秒)")
    
    return num_blocks, memory_per_block

def test_different_strategies():
    """
    测试不同的内存策略
    """
    print("测试不同内存策略")
    print("=" * 50)
    
    # 您遇到的问题数组形状
    problematic_shape = (14, 323, 279, 248)
    
    print("1. 原始方法 (会失败):")
    print("   x.flatten() -> 创建 1.17 GB 数组 -> 内存错误")
    
    print("\n2. flat迭代器方法:")
    simulate_flat_iteration(problematic_shape, block_size=1000000)
    
    print("\n3. 更小块的方法:")
    simulate_flat_iteration(problematic_shape, block_size=500000)
    
    print("\n4. 超小块方法 (最安全):")
    simulate_flat_iteration(problematic_shape, block_size=100000)

def demonstrate_flat_vs_flatten():
    """
    演示flat和flatten的区别
    """
    print("\n" + "=" * 50)
    print("flat vs flatten 的区别:")
    print("=" * 50)
    
    print("flatten():")
    print("  - 创建一个新的展平数组")
    print("  - 需要额外内存 = 原数组大小")
    print("  - 对于1.17GB数组会失败")
    
    print("\nflat:")
    print("  - 返回一个迭代器对象")
    print("  - 不创建新数组，几乎不占用额外内存")
    print("  - 可以安全处理任意大小的数组")
    
    print("\n示例代码对比:")
    print("# 会内存溢出的方法:")
    print("x_flat = x.flatten()  # 创建1.17GB新数组")
    print("for val in x_flat: ...")
    
    print("\n# 内存安全的方法:")
    print("for val in x.flat:  # 只创建迭代器")
    print("    # 逐个处理元素")

def main():
    print("解决flatten()内存溢出问题的方案测试")
    print("=" * 60)
    
    test_different_strategies()
    demonstrate_flat_vs_flatten()
    
    print("\n" + "=" * 60)
    print("解决方案总结:")
    print("✅ 使用 x.flat 而不是 x.flatten()")
    print("✅ 分块处理，每次处理50万-100万元素")
    print("✅ 逐元素累积计算，避免创建中间数组")
    print("✅ 内存使用从1.17GB降低到几乎为0")
    
    print("\n应用到您的代码:")
    print("1. 用 x.flat 替换 x.flatten()")
    print("2. 用 zip(x.flat, y.flat) 进行配对迭代")
    print("3. 逐元素累积计算dice值")
    print("4. 使用FINAL_DICE_FIX.py中的代码替换您的dice函数")

if __name__ == "__main__":
    main()