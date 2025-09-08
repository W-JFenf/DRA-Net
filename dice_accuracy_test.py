#!/usr/bin/env python3
"""
测试dice计算的准确性 - 验证分块方法与原始方法产生相同结果
"""

def create_test_data():
    """创建测试数据（模拟numpy数组）"""
    # 创建一个较小的测试数组来验证准确性
    # 使用Python列表模拟numpy数组
    size = 1000000  # 100万个元素
    
    # 创建测试数据：随机0和1
    import random
    random.seed(42)  # 固定种子保证可重复
    
    x = [random.choice([0, 1]) for _ in range(size)]
    y = [random.choice([0, 1]) for _ in range(size)]
    
    return x, y

def dice_original_method(x, y):
    """原始的dice计算方法（一次性处理所有数据）"""
    # 模拟 np.sum(np.sum(np.sum(x * y)))
    intersect = sum(x[i] * y[i] for i in range(len(x)))
    x_sum = sum(x)
    y_sum = sum(y)
    
    if y_sum == 0:
        return 0.0
    
    dice = (2 * intersect) / (x_sum + y_sum)
    return dice

def dice_chunked_method(x, y, chunk_size=100000):
    """分块的dice计算方法"""
    total_intersect = 0
    total_x_sum = 0
    total_y_sum = 0
    
    # 分块处理
    for i in range(0, len(x), chunk_size):
        end_idx = min(i + chunk_size, len(x))
        
        # 处理当前块
        chunk_intersect = 0
        chunk_x_sum = 0
        chunk_y_sum = 0
        
        for j in range(i, end_idx):
            chunk_intersect += x[j] * y[j]
            chunk_x_sum += x[j]
            chunk_y_sum += y[j]
        
        # 累加到总和
        total_intersect += chunk_intersect
        total_x_sum += chunk_x_sum
        total_y_sum += chunk_y_sum
    
    if total_y_sum == 0:
        return 0.0
    
    dice = (2 * total_intersect) / (total_x_sum + total_y_sum)
    return dice

def test_accuracy():
    """测试两种方法的准确性"""
    print("Dice计算准确性测试")
    print("=" * 50)
    
    # 创建测试数据
    print("创建测试数据...")
    x, y = create_test_data()
    print(f"数据大小: {len(x):,} 个元素")
    
    # 测试原始方法
    print("\n计算原始方法结果...")
    original_result = dice_original_method(x, y)
    print(f"原始方法结果: {original_result:.10f}")
    
    # 测试不同块大小的分块方法
    chunk_sizes = [10000, 50000, 100000, 200000, 500000]
    
    print("\n测试不同块大小的分块方法:")
    print("-" * 50)
    
    all_match = True
    for chunk_size in chunk_sizes:
        chunked_result = dice_chunked_method(x, y, chunk_size)
        difference = abs(original_result - chunked_result)
        
        print(f"块大小 {chunk_size:6,}: {chunked_result:.10f} (差异: {difference:.2e})")
        
        if difference > 1e-15:  # 允许极小的浮点误差
            all_match = False
    
    print("\n" + "=" * 50)
    if all_match:
        print("✅ 所有测试通过！分块方法与原始方法产生完全相同的结果")
        print("✅ 数学准确性得到验证")
    else:
        print("❌ 发现差异，需要检查实现")
    
    return all_match

def test_edge_cases():
    """测试边界情况"""
    print("\n边界情况测试")
    print("=" * 30)
    
    test_cases = [
        ("全零数组", [0] * 1000, [0] * 1000),
        ("全一数组", [1] * 1000, [1] * 1000),
        ("x全零", [0] * 1000, [1] * 1000),
        ("y全零", [1] * 1000, [0] * 1000),
        ("交替模式", [i % 2 for i in range(1000)], [(i+1) % 2 for i in range(1000)]),
    ]
    
    for name, x, y in test_cases:
        original = dice_original_method(x, y)
        chunked = dice_chunked_method(x, y, chunk_size=100)
        difference = abs(original - chunked)
        
        status = "✅" if difference < 1e-15 else "❌"
        print(f"{status} {name}: 原始={original:.6f}, 分块={chunked:.6f}, 差异={difference:.2e}")

def main():
    print("DICE计算准确性验证测试")
    print("验证分块方法是否会导致计算错误")
    print("=" * 60)
    
    # 主要准确性测试
    accuracy_ok = test_accuracy()
    
    # 边界情况测试
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("结论:")
    if accuracy_ok:
        print("🎯 分块方法在数学上完全等价于原始方法")
        print("🎯 不会导致dice计算错误")
        print("🎯 仅仅改变了内存使用方式，不改变数学结果")
        print("🎯 可以安全地用于替换原始实现")
    else:
        print("⚠️  需要进一步检查实现")
    
    print("\n原理解释:")
    print("• 求和运算满足结合律: (a+b+c+d) = (a+b) + (c+d)")
    print("• 分块处理只是改变了求和的顺序，不改变最终结果")
    print("• 浮点运算可能有极小误差(< 1e-15)，但在实践中可以忽略")

if __name__ == "__main__":
    main()