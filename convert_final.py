"""
DeepSpeed 检查点转换工具 - 最终修复版本
正确提取 module 中的模型权重
"""
import os
import sys
import torch
import glob

print("=" * 80)
print("DeepSpeed 检查点转换工具")
print("=" * 80)

checkpoint_dir = './checkpoints/best_model'
output_file = './checkpoints/pytorch_model.pt'

# 确保输出目录存在且有写权限
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 创建输出目录: {output_dir}")

# 测试写权限
try:
    test_file = os.path.join(output_dir, '.test_write')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print(f"✅ 输出目录有写权限")
except Exception as e:
    print(f"❌ 输出目录没有写权限: {e}")
    # 尝试使用当前目录
    output_file = './pytorch_model.pt'
    print(f"   改用当前目录: {output_file}")

# 步骤 1: 查找模型权重文件
print(f"\n📁 检查点目录: {checkpoint_dir}")

model_file_patterns = [
    'mp_rank_00_model_states.pt',
    '*model_states.pt',
]

model_file = None
for pattern in model_file_patterns:
    potential_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if potential_files:
        model_file = potential_files[0]
        break

if not model_file:
    print(f"❌ 错误: 未找到模型权重文件!")
    sys.exit(1)

print(f"✅ 找到模型权重文件: {os.path.basename(model_file)}")
file_size_mb = os.path.getsize(model_file) / (1024 * 1024)
print(f"   文件大小: {file_size_mb:.2f} MB")

# 步骤 2: 加载模型权重
print(f"\n📖 正在加载模型权重...")
try:
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    print(f"✅ 成功加载")

    # 步骤 3: 提取模型权重
    print(f"\n📊 检查点结构分析:")

    if isinstance(checkpoint, dict):
        print(f"   - 顶层键: {list(checkpoint.keys())}")

        # 检查是否包含 'module' 键
        if 'module' in checkpoint:
            print(f"   - 找到 'module' 键（DeepSpeed 包装的模型）")
            state_dict = checkpoint['module']

            if hasattr(state_dict, 'items'):  # 确保是字典类型
                print(f"   - 模型参数数量: {len(state_dict)}")

                # 显示前 10 个参数
                print(f"\n🔑 模型参数示例 (前 10 个):")
                for i, (key, value) in enumerate(list(state_dict.items())[:10]):
                    if hasattr(value, 'shape'):
                        print(f"   {i+1:2d}. {key:60s} -> {value.shape}")
                    else:
                        print(f"   {i+1:2d}. {key:60s} -> {type(value).__name__}")

                # 检查键名格式
                sample_keys = list(state_dict.keys())
                has_base_model = any(k.startswith('base_model.') for k in sample_keys)

                print(f"\n🔍 键名分析:")
                print(f"   - 包含 'base_model.' 前缀: {'是 ✅' if has_base_model else '否'}")

                if has_base_model:
                    print(f"   - 这是正确的 EfficientNetClassifier 权重格式")
                else:
                    print(f"   - ⚠️ 警告: 未检测到 'base_model.' 前缀")
                    print(f"   - 第一个键名: {sample_keys[0]}")

            else:
                print(f"❌ 错误: 'module' 不是字典类型: {type(state_dict)}")
                sys.exit(1)
        else:
            print(f"❌ 错误: 检查点不包含 'module' 键")
            print(f"   可用的键: {list(checkpoint.keys())}")
            sys.exit(1)
    else:
        print(f"❌ 错误: 检查点不是字典类型: {type(checkpoint)}")
        sys.exit(1)

    # 步骤 4: 保存为标准 PyTorch 格式
    print(f"\n💾 正在保存到: {output_file}")

    try:
        torch.save(state_dict, output_file)
        print(f"✅ 保存成功!")

        output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"   文件大小: {output_size_mb:.2f} MB")

    except Exception as save_error:
        print(f"❌ 保存失败: {save_error}")
        # 尝试保存到当前目录
        alt_output = './pytorch_model.pt'
        print(f"\n尝试保存到当前目录: {alt_output}")
        torch.save(state_dict, alt_output)
        output_file = alt_output
        print(f"✅ 成功保存到: {output_file}")

    # 步骤 5: 验证
    print(f"\n🔍 验证保存的文件...")
    verify_dict = torch.load(output_file, map_location='cpu', weights_only=False)
    print(f"   ✅ 验证通过，包含 {len(verify_dict)} 个参数")

    # 显示示例键名
    verify_keys = list(verify_dict.keys())[:5]
    print(f"\n   示例键名:")
    for key in verify_keys:
        print(f"      - {key}")

    # 成功提示
    print("\n" + "=" * 80)
    print("✅ 转换成功!")
    print("=" * 80)

    print(f"\n📝 接下来的步骤:")
    print(f"\n1. 修改 predict_pytorch.py (第 11 行):")
    print(f"   model_path = '{output_file}'")

    print(f"\n2. 修改 demo_pytorch.py (第 64 行):")
    print(f"   self.model_path = '{output_file}'")

    print(f"\n3. 简化加载代码为:")
    print(f"   state_dict = torch.load('{output_file}', map_location=device)")
    print(f"   model.load_state_dict(state_dict)")

    print(f"\n4. 重新测试:")
    print(f"   python predict_pytorch.py")
    print(f"\n   ✅ 预期: 置信度应该 >80%")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ 转换失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
