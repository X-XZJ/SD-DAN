import os
import sys
from typing import Dict, Tuple, List

import torch
import torch.nn as nn


class ComplexityAnalyzer:
    """
    模型复杂度分析工具（不依赖fvcore）
    
    支持计算：
    - 参数量 (Parameters)
    - 计算量 (FLOPs) - 基于经验公式估计
    - 内存占用 (Memory)
    - 各模块复杂度对比
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 800, 1333)):
        """
        初始化分析器
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状 (batch_size, channels, height, width)
        """
        self.model = model
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        
    def count_parameters(self) -> Dict[str, float]:
        """
        计算模型参数量
        
        Returns:
            {
                'total': 总参数数,
                'trainable': 可训练参数数,
                'non_trainable': 不可训练参数数,
                'total_M': 总参数(百万),
                'trainable_M': 可训练参数(百万),
                'non_trainable_M': 不可训练参数(百万)
            }
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for param in self.model.parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            else:
                non_trainable_params += num_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params,
            'total_M': total_params / 1e6,
            'trainable_M': trainable_params / 1e6,
            'non_trainable_M': non_trainable_params / 1e6,
        }
    
    def estimate_flops_by_conv_linear(self) -> Dict[str, float]:
        """
        通过卷积和线性层估计FLOPs
        
        这是一个更精确的估计方法，遍历模型的所有层
        并根据具体的操作类型计算FLOPs
        
        Returns:
            {
                'flops': 总FLOPs数,
                'params': 参数数,
                'flops_G': FLOPs(十亿),
                'params_M': 参数(百万),
                'ratio': FLOPs/参数比
            }
        """
        params = self.count_parameters()
        total_params = params['total']
        
        flops = 0
        batch_size, channels, height, width = self.input_shape
        
        # 遍历所有模块，估计FLOPs
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d FLOPs计算
                # FLOPs = kernel_height * kernel_width * in_channels * out_channels * output_height * output_width / groups
                output_h = (height - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0] + 1
                output_w = (width - module.kernel_size[1] + 2 * module.padding[1]) // module.stride[1] + 1
                
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
                output_flops = output_h * output_w * module.out_channels
                module_flops = kernel_flops * output_flops * batch_size
                
                flops += module_flops
                
                # 更新高度和宽度用于后续层
                height = output_h
                width = output_w
                channels = module.out_channels
                
            elif isinstance(module, nn.Linear):
                # Linear FLOPs = in_features * out_features * batch_size
                module_flops = module.in_features * module.out_features * batch_size
                flops += module_flops
            
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # BatchNorm FLOPs ≈ 2 * number_of_elements
                if isinstance(module, nn.BatchNorm2d):
                    num_elements = batch_size * module.num_features * height * width
                else:
                    num_elements = batch_size * module.num_features
                flops += 2 * num_elements
        
        flops_G = flops / 1e9
        params_M = total_params / 1e6
        
        return {
            'flops': flops,
            'params': total_params,
            'flops_G': flops_G,
            'params_M': params_M,
            'ratio': flops_G / params_M if params_M > 0 else 0,
        }
    
    def estimate_flops_by_formula(self) -> Dict[str, float]:
        """
        使用经验公式估计FLOPs（用于Transformer模型）
        
        对于Transformer模型：
        - FLOPs ≈ 参数数 × 2 × 序列长度系数
        - 对于DETR（Vision Transformer）：系数通常为2-3
        
        Returns:
            {
                'flops': 总FLOPs数,
                'params': 参数数,
                'flops_G': FLOPs(十亿),
                'params_M': 参数(百万),
                'ratio': FLOPs/参数比,
                'note': 计算方法说明
            }
        """
        params = self.count_parameters()
        total_params = params['total']
        
        # 计算特征图大小
        _, _, height, width = self.input_shape
        # 特征图经过ResNet50 backbone后的尺寸（通常下采样32倍）
        feature_height = height // 32
        feature_width = width // 32
        feature_size = feature_height * feature_width
        
        # DETR中transformer处理的序列长度 = 特征图像素数 + 查询数量
        # 通常查询数量为300-900
        query_size = 900  # 可配置
        sequence_length = feature_size + query_size
        
        # 经验公式：Transformer的FLOPs ≈ params × 2 × sequence_length / 1000
        # 这里除以1000是因为并不是所有参数都参与每次前向传播
        estimated_flops = total_params * 2 * (sequence_length / 1000)
        
        flops_G = estimated_flops / 1e9
        params_M = total_params / 1e6
        
        return {
            'flops': estimated_flops,
            'params': total_params,
            'flops_G': flops_G,
            'params_M': params_M,
            'ratio': flops_G / params_M if params_M > 0 else 0,
            'note': f'(经验公式估计，特征图尺寸: {feature_height}×{feature_width}, 序列长度: {sequence_length})',
            'sequence_length': sequence_length,
            'feature_size': feature_size,
        }
    
    def count_flops(self) -> Dict[str, float]:
        """
        计算模型FLOPs（尝试多种方法）
        
        Returns:
            {
                'flops': 总FLOPs数,
                'params': 参数数,
                'flops_G': FLOPs(十亿),
                'params_M': 参数(百万),
                'ratio': FLOPs/参数比
            }
        """
        try:
            # 首先尝试基于卷积和线性层的精确计算
            result = self.estimate_flops_by_conv_linear()
            result['method'] = 'Conv/Linear-based calculation'
            return result
        except Exception as e:
            print(f"⚠️  基于卷积层的计算失败 ({e})，使用公式估计...")
            try:
                return self.estimate_flops_by_formula()
            except Exception as e2:
                print(f"⚠️  公式估计也失败 ({e2})，使用简化估计...")
                return self._estimate_flops_simple()
    
    def _estimate_flops_simple(self) -> Dict[str, float]:
        """
        最简单的FLOPs估计
        """
        params = self.count_parameters()
        total_params = params['total']
        
        # 保守估计：Transformer的FLOPs约为参数数的2.5倍
        estimated_flops = total_params * 2.5
        flops_G = estimated_flops / 1e9
        params_M = total_params / 1e6
        
        return {
            'flops': estimated_flops,
            'params': total_params,
            'flops_G': flops_G,
            'params_M': params_M,
            'ratio': flops_G / params_M if params_M > 0 else 0,
            'note': '(简化估计：FLOPs = params × 2.5)',
        }
    
    def analyze_by_module(self) -> Dict[str, Dict]:
        """
        按模块统计参数量
        
        Returns:
            各模块的详细统计信息
        """
        module_stats = {}
        
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if params > 0:
                    module_stats[name] = {
                        'params': params,
                        'params_M': params / 1e6,
                        'type': module.__class__.__name__,
                    }
        
        return module_stats
    
    def get_memory_estimate(self) -> Dict[str, float]:
        """
        估计模型内存占用
        
        内存包括：
        1. 参数内存 = 参数数 × 字节/参数
        2. 激活值内存 = 隐层特征激活值占用内存
        3. 梯度内存 = 参数梯度占用内存（训练时）
        
        Returns:
            {
                'params_memory_MB': 参数内存(MB),
                'activation_memory_MB': 激活值内存(MB),
                'total_memory_MB': 总内存(MB)
            }
        """
        params = self.count_parameters()
        
        # 1. 参数内存 (假设使用float32，每个参数4字节)
        params_memory_bytes = params['total'] * 4
        params_memory_mb = params_memory_bytes / (1024 ** 2)
        
        # 2. 激活值内存估计
        # 对于DETR，主要的激活来自：
        # - 图像特征 (batch_size × channels × height × width)
        # - Transformer特征 (batch_size × sequence_length × embed_dim)
        
        batch_size, channels, height, width = self.input_shape
        
        # 图像特征激活（经过backbone后）
        feature_height = height // 32
        feature_width = width // 32
        embed_dim = 256  # 从配置中推断
        image_activation_elements = batch_size * embed_dim * feature_height * feature_width
        
        # Transformer序列激活
        query_size = 900
        sequence_length = feature_height * feature_width + query_size
        transformer_activation_elements = batch_size * sequence_length * embed_dim
        
        # 总激活元素数
        total_activation_elements = image_activation_elements + transformer_activation_elements
        
        # 激活内存 (float32, 4字节)
        activation_memory_mb = (total_activation_elements * 4) / (1024 ** 2)
        
        # 3. 梯度内存（训练时，同参数内存）
        gradient_memory_mb = params_memory_mb
        
        # 总内存
        total_memory_mb = params_memory_mb + activation_memory_mb + gradient_memory_mb
        
        return {
            'params_memory_MB': params_memory_mb,
            'activation_memory_MB': activation_memory_mb,
            'gradient_memory_MB': gradient_memory_mb,
            'total_memory_MB': total_memory_mb,
            'params_memory_GB': params_memory_mb / 1024,
            'activation_memory_GB': activation_memory_mb / 1024,
            'total_memory_GB': total_memory_mb / 1024,
        }
    
    def print_summary(self):
        """打印完整的复杂度分析报告"""
        print("\n" + "="*90)
        print("🔍 模型复杂度分析报告")
        print("="*90)
        
        # 1. 参数量统计
        print("\n📊 参数量统计:")
        print("-" * 90)
        params_info = self.count_parameters()
        print(f"  总参数数:        {params_info['total']:>15,} ({params_info['total_M']:>10.2f}M)")
        print(f"  可训练参数:      {params_info['trainable']:>15,} ({params_info['trainable_M']:>10.2f}M)")
        print(f"  不可训练参数:    {params_info['non_trainable']:>15,} ({params_info['non_trainable_M']:>10.2f}M)")
        
        # 2. FLOPs统计
        print("\n⚡ 计算量统计 (FLOPs):")
        print("-" * 90)
        flops_info = self.count_flops()
        print(f"  总FLOPs:         {flops_info['flops_G']:>15.2f}G")
        print(f"  FLOPs/参数比:    {flops_info['ratio']:>15.2f} (GFLOPs/M)")
        if 'method' in flops_info:
            print(f"  计算方法:        {flops_info['method']}")
        if 'note' in flops_info:
            print(f"  注:              {flops_info['note']}")
        if 'sequence_length' in flops_info:
            print(f"  序列长度:        {flops_info['sequence_length']:>15}")
            print(f"  特征图大小:      {flops_info['feature_size']:>15}")
        
        # 3. 内存占用
        print("\n💾 内存占用估计:")
        print("-" * 90)
        memory_info = self.get_memory_estimate()
        print(f"  参数内存:        {memory_info['params_memory_MB']:>15.2f}MB ({memory_info['params_memory_GB']:>8.2f}GB)")
        print(f"  激活值内存:      {memory_info['activation_memory_MB']:>15.2f}MB ({memory_info['activation_memory_GB']:>8.2f}GB)")
        print(f"  梯度内存:        {memory_info['gradient_memory_MB']:>15.2f}MB")
        print(f"  总内存占用:      {memory_info['total_memory_MB']:>15.2f}MB ({memory_info['total_memory_GB']:>8.2f}GB)")
        
        # 4. 模块级别统计（显示top-15）
        print("\n🔧 模块级别统计 (Top-15 按参数量):")
        print("-" * 90)
        module_stats = self.analyze_by_module()
        sorted_modules = sorted(
            module_stats.items(),
            key=lambda x: x[1]['params'],
            reverse=True
        )[:15]
        
        print(f"{'模块名称':<55} {'参数数':>15} {'参数量(M)':>12}")
        print("-" * 90)
        for name, stats in sorted_modules:
            # 截断长名称
            short_name = name if len(name) <= 55 else "..." + name[-52:]
            print(f"{short_name:<55} {stats['params']:>15,} {stats['params_M']:>12.2f}M")
        
        print("\n" + "="*90 + "\n")


class ModelComparisonAnalyzer:
    """模型对比分析工具"""
    
    def __init__(self, models_dict: Dict[str, nn.Module], input_shape: Tuple = (1, 3, 800, 1333)):
        """
        Args:
            models_dict: {模型名称: 模型对象}
            input_shape: 输入形状
        """
        self.models_dict = models_dict
        self.input_shape = input_shape
        self.analyzers = {
            name: ComplexityAnalyzer(model, input_shape)
            for name, model in models_dict.items()
        }
    
    def compare(self):
        """对比多个模型的复杂度"""
        print("\n" + "="*110)
        print("🔬 模型复杂度对比分析")
        print("="*110)
        
        results = {}
        for model_name, analyzer in self.analyzers.items():
            params = analyzer.count_parameters()
            flops = analyzer.count_flops()
            memory = analyzer.get_memory_estimate()
            
            results[model_name] = {
                'params_M': params['total_M'],
                'flops_G': flops['flops_G'],
                'ratio': flops['ratio'],
                'memory_MB': memory['total_memory_MB'],
            }
        
        # 打印对比表格
        print(f"\n{'模型名称':<30} {'参数(M)':>15} {'FLOPs(G)':>15} {'比率':>15} {'内存(MB)':>15}")
        print("-" * 110)
        
        for model_name, stats in results.items():
            print(
                f"{model_name:<30} "
                f"{stats['params_M']:>15.2f} "
                f"{stats['flops_G']:>15.2f} "
                f"{stats['ratio']:>15.2f} "
                f"{stats['memory_MB']:>15.2f}"
            )
        
        # 计算增量
        print("\n📈 增量分析:")
        print("-" * 110)
        base_model = list(results.keys())[0]
        base_stats = results[base_model]
        
        for model_name, stats in list(results.items())[1:]:
            param_increase = (stats['params_M'] - base_stats['params_M']) / base_stats['params_M'] * 100
            flops_increase = (stats['flops_G'] - base_stats['flops_G']) / base_stats['flops_G'] * 100
            memory_increase = (stats['memory_MB'] - base_stats['memory_MB']) / base_stats['memory_MB'] * 100
            
            print(f"\n{model_name} vs {base_model}:")
            print(f"  参数增长:  {param_increase:>+.2f}%")
            print(f"  FLOPs增长: {flops_increase:>+.2f}%")
            print(f"  内存增长:  {memory_increase:>+.2f}%")
        
        print("\n" + "="*110 + "\n")


# 快速计算函数
def quick_stats(model: nn.Module, input_shape: Tuple = (1, 3, 800, 1333)):
    """快速获取模型统计信息"""
    analyzer = ComplexityAnalyzer(model, input_shape)
    analyzer.print_summary()
    return {
        'params': analyzer.count_parameters(),
        'flops': analyzer.count_flops(),
        'memory': analyzer.get_memory_estimate(),
    }