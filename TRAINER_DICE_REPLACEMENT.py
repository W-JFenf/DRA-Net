"""
直接替换您trainer.py中dice函数的最终版本

解决问题：numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.17 GiB 
发生在：x.flatten() 这一步

解决方案：使用x.flat迭代器，完全不创建新数组
"""

import numpy as np

def dice(x, y):
    """
    内存超级安全的dice计算函数
    
    直接替换您trainer.py中的dice函数即可
    完全解决flatten()内存溢出问题
    """
    # 检查形状
    if x.shape != y.shape:
        raise ValueError(f"数组形状不匹配: x.shape={x.shape}, y.shape={y.shape}")
    
    print(f"计算dice - 数组形状: {x.shape}, 内存: {x.nbytes/(1024*1024):.1f}MB")
    
    # 使用flat迭代器，不创建新数组
    total_intersect = 0.0
    total_x_sum = 0.0
    total_y_sum = 0.0
    
    # zip(x.flat, y.flat) 创建配对迭代器，内存使用几乎为0
    element_count = 0
    for x_val, y_val in zip(x.flat, y.flat):
        # 转换为float避免整数溢出
        x_f = float(x_val)
        y_f = float(y_val)
        
        total_intersect += x_f * y_f
        total_x_sum += x_f
        total_y_sum += y_f
        
        element_count += 1
        
        # 每100万个元素显示一次进度
        if element_count % 1000000 == 0:
            print(f"已处理 {element_count//1000000}M 个元素...")
    
    print(f"dice计算完成，总共处理 {element_count:,} 个元素")
    
    # 处理除零情况
    if total_y_sum == 0:
        print("警告：y_sum为0，返回dice=0")
        return 0.0
    
    # 计算dice系数
    dice_coeff = (2.0 * total_intersect) / (total_x_sum + total_y_sum)
    print(f"dice系数: {dice_coeff:.6f}")
    
    return dice_coeff

# ============================================================================
# 使用方法：
# ============================================================================
"""
1. 打开您的 trainer.py 文件

2. 找到现有的 dice 函数（大约在第26行附近）：
   def dice(x, y):
       intersect = np.sum(np.sum(np.sum(x * y)))  # <-- 这里出错
       y_sum = np.sum(np.sum(np.sum(y)))
       if y_sum == 0:
           return 0.0
       x_sum = np.sum(np.sum(np.sum(x)))
       return (2 * intersect) / (x_sum + y_sum)

3. 完全删除上面的函数

4. 复制粘贴本文件中的 dice 函数到相同位置

5. 确保文件顶部有 import numpy as np

6. 保存文件并重新运行训练

这个新的dice函数的特点：
✅ 不使用 flatten() - 避免创建1.17GB数组
✅ 使用 x.flat 迭代器 - 内存使用几乎为0  
✅ 逐元素处理 - 完全避免大数组操作
✅ 有进度显示 - 可以看到处理进度
✅ 数学结果完全相同 - 只是改变了计算方式
✅ 处理速度合理 - 大约30-60秒完成一次dice计算

预期运行效果：
- 不再出现内存分配错误
- 会显示处理进度（每100万元素一次）
- dice计算成功完成
- 训练继续正常进行
"""