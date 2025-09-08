"""
超级内存高效的dice计算 - 解决连flatten()都会内存溢出的问题
"""
import numpy as np

def dice_ultra_memory_efficient(x, y):
    """
    超级内存高效的dice计算，完全不创建新的数组
    适用于连flatten()都会内存溢出的极端情况
    
    Args:
        x: 预测数组 (任意维度的numpy数组)
        y: 真实标签数组 (与x相同形状)
    
    Returns:
        dice系数 (float)
    """
    # 检查形状是否匹配
    if x.shape != y.shape:
        raise ValueError(f"数组形状不匹配: x.shape={x.shape}, y.shape={y.shape}")
    
    # 初始化累积变量
    total_intersect = 0.0
    total_x_sum = 0.0  
    total_y_sum = 0.0
    
    # 使用numpy的nditer进行内存高效的遍历
    # nditer不会创建新数组，而是直接遍历现有数组
    it_x = np.nditer(x, flags=['multi_index', 'refs_ok'])
    it_y = np.nditer(y, flags=['multi_index', 'refs_ok'])
    
    # 逐元素处理，完全避免创建中间数组
    while not it_x.finished:
        x_val = float(it_x[0])
        y_val = float(it_y[0])
        
        # 累积计算
        total_intersect += x_val * y_val
        total_x_sum += x_val
        total_y_sum += y_val
        
        # 移动到下一个元素
        it_x.iternext()
        it_y.iternext()
    
    # 处理除零情况
    if total_y_sum == 0:
        return 0.0
    
    # 计算dice系数
    dice_coeff = (2.0 * total_intersect) / (total_x_sum + total_y_sum)
    return dice_coeff


def dice_block_iterator(x, y, block_shape=(100, 100, 100)):
    """
    使用块迭代器的dice计算，按块处理数据
    适用于大型多维数组
    
    Args:
        x: 预测数组
        y: 真实标签数组  
        block_shape: 每个块的形状
    
    Returns:
        dice系数 (float)
    """
    if x.shape != y.shape:
        raise ValueError(f"数组形状不匹配: x.shape={x.shape}, y.shape={y.shape}")
    
    total_intersect = 0.0
    total_x_sum = 0.0
    total_y_sum = 0.0
    
    # 计算需要多少个块
    shape = x.shape
    
    # 为了避免内存问题，我们按维度分块处理
    if len(shape) == 4:  # (C, H, W, D)
        c_size, h_size, w_size, d_size = shape
        bh, bw, bd = block_shape
        
        for c in range(c_size):
            for h_start in range(0, h_size, bh):
                for w_start in range(0, w_size, bw):
                    for d_start in range(0, d_size, bd):
                        # 计算当前块的边界
                        h_end = min(h_start + bh, h_size)
                        w_end = min(w_start + bw, w_size)
                        d_end = min(d_start + bd, d_size)
                        
                        # 提取当前块（这里只创建视图，不复制数据）
                        x_block = x[c, h_start:h_end, w_start:w_end, d_start:d_end]
                        y_block = y[c, h_start:h_end, w_start:w_end, d_start:d_end]
                        
                        # 使用numpy的内置函数，它们通常更内存高效
                        block_intersect = np.sum(x_block * y_block)
                        block_x_sum = np.sum(x_block)
                        block_y_sum = np.sum(y_block)
                        
                        # 累积结果
                        total_intersect += float(block_intersect)
                        total_x_sum += float(block_x_sum)
                        total_y_sum += float(block_y_sum)
    
    # 处理除零情况
    if total_y_sum == 0:
        return 0.0
    
    # 计算dice系数
    dice_coeff = (2.0 * total_intersect) / (total_x_sum + total_y_sum)
    return dice_coeff


def dice_minimal_memory(x, y):
    """
    最小内存使用的dice计算
    使用最保守的方法，确保在极低内存环境下也能工作
    """
    if x.shape != y.shape:
        raise ValueError(f"数组形状不匹配: x.shape={x.shape}, y.shape={y.shape}")
    
    # 获取数组的总元素数，但不创建新数组
    total_elements = x.size
    
    # 使用最小的内存块大小
    chunk_size = min(1000, total_elements)  # 最多1000个元素一次
    
    total_intersect = 0.0
    total_x_sum = 0.0
    total_y_sum = 0.0
    
    # 使用ravel()的flat属性，它返回一个迭代器而不是新数组
    x_flat = x.flat
    y_flat = y.flat
    
    # 分批处理
    processed = 0
    while processed < total_elements:
        # 确定当前批次的大小
        current_chunk = min(chunk_size, total_elements - processed)
        
        # 处理当前批次
        chunk_intersect = 0.0
        chunk_x_sum = 0.0
        chunk_y_sum = 0.0
        
        for _ in range(current_chunk):
            try:
                x_val = next(x_flat)
                y_val = next(y_flat)
                
                chunk_intersect += float(x_val) * float(y_val)
                chunk_x_sum += float(x_val)
                chunk_y_sum += float(y_val)
                
            except StopIteration:
                break
        
        # 累积结果
        total_intersect += chunk_intersect
        total_x_sum += chunk_x_sum
        total_y_sum += chunk_y_sum
        
        processed += current_chunk
    
    # 处理除零情况
    if total_y_sum == 0:
        return 0.0
    
    # 计算dice系数
    dice_coeff = (2.0 * total_intersect) / (total_x_sum + total_y_sum)
    return dice_coeff


# 最终的dice函数 - 多层次回退策略
def dice(x, y):
    """
    多层次回退的dice计算
    从最快的方法开始尝试，如果内存不足就回退到更保守的方法
    """
    try:
        # 方法1: 尝试块迭代器方法（相对较快）
        return dice_block_iterator(x, y, block_shape=(50, 50, 50))
    except MemoryError:
        print("警告：块迭代器方法内存不足，回退到超级节省内存方法")
        
        try:
            # 方法2: 尝试超级内存高效方法
            return dice_ultra_memory_efficient(x, y)
        except MemoryError:
            print("警告：超级内存高效方法仍然内存不足，使用最小内存方法")
            
            # 方法3: 最后的回退 - 最小内存方法
            return dice_minimal_memory(x, y)


# 专门针对您的错误的修复版本
def dice_fix_for_flatten_error(x, y):
    """
    专门修复flatten()内存错误的版本
    """
    print(f"处理数组形状: x.shape={x.shape}, y.shape={y.shape}")
    print(f"数组数据类型: x.dtype={x.dtype}, y.dtype={y.dtype}")
    
    # 计算内存使用情况
    x_memory_mb = x.nbytes / (1024 * 1024)
    y_memory_mb = y.nbytes / (1024 * 1024)
    print(f"数组内存使用: x={x_memory_mb:.1f}MB, y={y_memory_mb:.1f}MB")
    
    # 使用最保守的方法
    return dice_minimal_memory(x, y)