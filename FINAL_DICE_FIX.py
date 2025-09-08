"""
最终的DICE修复方案 - 解决flatten()内存溢出问题

直接替换您trainer.py中的dice相关函数
"""

import numpy as np

def dice_memory_efficient(x, y, chunk_size=500000):
    """
    这个版本已经不适用了，因为连flatten()都会内存溢出
    保留这里仅供参考
    """
    # 这个方法在您的情况下会失败，因为x.flatten()本身就需要1.17GB内存
    pass

def dice_ultra_safe(x, y):
    """
    超级安全的dice计算 - 完全不创建新数组
    适用于连flatten()都会内存溢出的极端情况
    
    使用numpy的flat属性，它返回迭代器而不是新数组
    """
    if x.shape != y.shape:
        raise ValueError(f"数组形状不匹配: x.shape={x.shape}, y.shape={y.shape}")
    
    print(f"开始处理数组 - 形状: {x.shape}, 内存: {x.nbytes/(1024*1024):.1f}MB")
    
    # 使用flat属性获取迭代器，不创建新数组
    x_iter = x.flat
    y_iter = y.flat
    
    total_intersect = 0.0
    total_x_sum = 0.0
    total_y_sum = 0.0
    
    # 逐元素处理，完全避免创建中间数组
    element_count = 0
    for x_val, y_val in zip(x_iter, y_iter):
        total_intersect += float(x_val) * float(y_val)
        total_x_sum += float(x_val)
        total_y_sum += float(y_val)
        
        element_count += 1
        # 每处理100万个元素打印一次进度
        if element_count % 1000000 == 0:
            print(f"已处理 {element_count/1000000:.1f}M 个元素...")
    
    print(f"处理完成，总共 {element_count} 个元素")
    
    # 处理除零情况
    if total_y_sum == 0:
        return 0.0
    
    # 计算dice系数
    dice_coeff = (2.0 * total_intersect) / (total_x_sum + total_y_sum)
    return dice_coeff

def dice_block_safe(x, y, block_size=1000000):
    """
    基于块的安全dice计算
    将大数组分成小块处理，每次只处理一小部分
    """
    if x.shape != y.shape:
        raise ValueError(f"数组形状不匹配: x.shape={x.shape}, y.shape={y.shape}")
    
    total_elements = x.size
    print(f"总元素数: {total_elements:,}, 将分成 {(total_elements + block_size - 1) // block_size} 个块处理")
    
    total_intersect = 0.0
    total_x_sum = 0.0
    total_y_sum = 0.0
    
    # 使用numpy的flat迭代器
    x_flat = x.flat
    y_flat = y.flat
    
    processed = 0
    block_num = 1
    
    while processed < total_elements:
        # 计算当前块的大小
        current_block_size = min(block_size, total_elements - processed)
        
        print(f"处理第 {block_num} 块 ({current_block_size:,} 个元素)...")
        
        # 处理当前块
        block_intersect = 0.0
        block_x_sum = 0.0
        block_y_sum = 0.0
        
        for _ in range(current_block_size):
            x_val = next(x_flat)
            y_val = next(y_flat)
            
            block_intersect += float(x_val) * float(y_val)
            block_x_sum += float(x_val)
            block_y_sum += float(y_val)
        
        # 累积结果
        total_intersect += block_intersect
        total_x_sum += block_x_sum
        total_y_sum += block_y_sum
        
        processed += current_block_size
        block_num += 1
    
    # 处理除零情况
    if total_y_sum == 0:
        return 0.0
    
    # 计算dice系数
    dice_coeff = (2.0 * total_intersect) / (total_x_sum + total_y_sum)
    return dice_coeff

def dice(x, y):
    """
    最终的dice函数 - 直接替换您trainer.py中的dice函数
    
    这个函数会自动选择最安全的计算方法
    """
    try:
        # 首先尝试相对较快的块方法
        return dice_block_safe(x, y, block_size=500000)  # 500K元素 ≈ 2MB
    except MemoryError as e:
        print(f"块方法失败: {e}")
        print("回退到超级安全方法...")
        
        # 如果还是失败，使用最安全的逐元素方法
        return dice_ultra_safe(x, y)

# ============================================================================
# 使用说明
# ============================================================================
"""
将您的trainer.py中的dice函数完全替换为上面的代码：

1. 删除原来的dice函数和dice_memory_efficient函数

2. 复制上面的dice_ultra_safe、dice_block_safe和dice函数到您的trainer.py文件中

3. 确保import numpy as np在文件顶部

完整的替换代码：
```python
def dice_ultra_safe(x, y):
    # ... (复制上面的完整函数)

def dice_block_safe(x, y, block_size=1000000):
    # ... (复制上面的完整函数)

def dice(x, y):
    # ... (复制上面的完整函数)
```

这个版本的特点：
- 完全不使用flatten()，避免创建1.17GB的展平数组
- 使用numpy的flat迭代器，只创建迭代器不创建新数组
- 分块处理，每次最多处理500K个元素（约2MB）
- 有进度显示，您可以看到处理进度
- 多层次回退，如果一种方法失败会自动尝试更安全的方法
"""