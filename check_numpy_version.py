"""
检查numpy版本和相关信息的脚本
可以在您的环境中运行来获取具体信息
"""

def check_numpy_info():
    """检查numpy版本和配置信息"""
    print("NumPy环境检查")
    print("=" * 40)
    
    try:
        import numpy as np
        print(f"✅ NumPy版本: {np.__version__}")
        
        # 检查numpy配置
        print(f"✅ NumPy安装路径: {np.__file__}")
        
        # 检查BLAS库信息
        try:
            config = np.__config__
            print(f"✅ NumPy配置可用")
        except:
            print("⚠️  无法获取NumPy配置信息")
        
        # 检查可用内存相关设置
        print(f"✅ NumPy数据类型大小:")
        print(f"   - float32: {np.dtype('float32').itemsize} bytes")
        print(f"   - float64: {np.dtype('float64').itemsize} bytes")
        
        # 测试小数组创建
        test_array = np.ones((100, 100), dtype=np.float32)
        print(f"✅ 小数组创建测试通过: {test_array.shape}")
        
        # 计算您的问题数组需要的内存
        shape = (14, 323, 279, 248)
        total_elements = np.prod(shape)
        memory_gb = (total_elements * 4) / (1024**3)  # float32 = 4 bytes
        
        print(f"\n您的数组信息:")
        print(f"- 形状: {shape}")
        print(f"- 总元素: {total_elements:,}")
        print(f"- 需要内存: {memory_gb:.2f} GB")
        
    except ImportError:
        print("❌ NumPy未安装")
        return False
    except Exception as e:
        print(f"❌ 检查NumPy时出错: {e}")
        return False
    
    return True

def analyze_version_impact():
    """分析版本影响"""
    print("\n版本影响分析")
    print("=" * 40)
    
    try:
        import numpy as np
        version = np.__version__
        major, minor = map(int, version.split('.')[:2])
        
        print(f"当前版本: {version}")
        
        if major == 1:
            if minor < 17:
                print("📊 版本分析: 较老版本 (< 1.17)")
                print("   - 内存错误可能显示为简单的 MemoryError")
                print("   - 内存分配相对宽松")
                
            elif minor < 20:
                print("📊 版本分析: 中等版本 (1.17-1.19)")
                print("   - 开始出现详细的 _ArrayMemoryError")
                print("   - 内存检查有所加强")
                
            else:
                print("📊 版本分析: 较新版本 (1.20+)")
                print("   - 严格的内存检查")
                print("   - 详细的错误信息")
                print("   - 这可能是您遇到问题的原因")
        
        print(f"\n基于您的错误信息:")
        print(f"numpy.core._exceptions._ArrayMemoryError")
        print(f"→ 确认您使用的是numpy 1.17+版本")
        
    except Exception as e:
        print(f"无法分析版本: {e}")

def version_solutions():
    """版本相关解决方案"""
    print("\n解决方案选择")
    print("=" * 40)
    
    print("选项1: 降级numpy (不推荐)")
    print("命令: pip install numpy==1.16.6")
    print("风险:")
    print("  ❌ 可能破坏MONAI等依赖")
    print("  ❌ 失去性能改进")
    print("  ❌ 安全漏洞")
    
    print("\n选项2: 升级到最新版本 (可以尝试)")
    print("命令: pip install --upgrade numpy")
    print("优点:")
    print("  ✅ 可能有内存优化")
    print("  ✅ 最新功能和修复")
    print("风险:")
    print("  ⚠️  可能仍有同样问题")
    
    print("\n选项3: 使用内存安全方案 (强烈推荐)")
    print("方法: 替换dice函数实现")
    print("优点:")
    print("  ✅ 与版本无关")
    print("  ✅ 从根本上解决问题")
    print("  ✅ 不影响其他功能")
    print("  ✅ 数学结果完全相同")

def main():
    numpy_available = check_numpy_info()
    
    if numpy_available:
        analyze_version_impact()
    
    version_solutions()
    
    print("\n" + "=" * 40)
    print("建议:")
    print("1. 先尝试我们的内存安全dice函数")
    print("2. 如果还有问题，考虑升级numpy")
    print("3. 最后才考虑降级（不推荐）")

if __name__ == "__main__":
    main()