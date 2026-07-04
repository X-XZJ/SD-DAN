import torch
import copy
import time
from typing import Dict, Any, Tuple

# 检查可用的FLOPs计算库
try:
    import thop
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️ thop库不可用，将使用估算方法")


class ModelAnalyzer:
    """模型复杂度分析器"""
    
    def __init__(self, model, input_size=(3, 800, 1333), device='cuda'):
        self.model = model
        self.input_size = input_size
        self.device = device
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型的详细信息"""
        info = {}
        
        # 参数统计
        param_info = self._analyze_parameters()
        info.update(param_info)
        
        # FLOPs计算
        flops_info = self._calculate_flops()
        info.update(flops_info)
        
        # 模型内存占用
        memory_info = self._estimate_memory()
        info.update(memory_info)
        
        # # 推理速度测试
        # try:
        #     speed_info = self._measure_inference_speed()
        #     info.update(speed_info)
        # except Exception as e:
        #     print(f"推理速度测试失败: {e}")
        #     info.update({
        #         'inference_time_ms': 0.0,
        #         'fps': 0.0,
        #         'throughput': 0.0
        #     })
        
        return info
    
    def _analyze_parameters(self) -> Dict[str, Any]:
        """分析模型参数"""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for param in self.model.parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            else:
                frozen_params += param_count
        
        return {
            'params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6,
            'frozen_params_M': frozen_params / 1e6
        }
    
    def _calculate_flops(self) -> Dict[str, Any]:
        """FLOPs计算方法 - 优先使用thop，失败时使用估算"""
        flops_info = {
            'flops': 0,
            'flops_G': 0.0,
            'flops_method': '未计算',
            'mac': 0,
            'mac_G': 0.0
        }
        
        # 保存原始状态
        original_training = self.model.training
        original_device = next(self.model.parameters()).device
        
        try:
            self.model.eval()
            
            # 尝试使用thop
            if THOP_AVAILABLE:
                try:
                    print("正在使用thop计算FLOPs...")
                    
                    # 确保模型和输入在同一设备
                    target_device = original_device
                    model_copy = copy.deepcopy(self.model).to(target_device)
                    dummy_input = torch.randn(1, *self.input_size).to(target_device)
                    
                 
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("thop计算超时")
                    
                   
                    
                    
                    with torch.no_grad():
                        flops, params = thop.profile(model_copy, inputs=(dummy_input,), verbose=False)
                    

                    flops_info.update({
                        'flops': flops,
                        'flops_G': flops / 1e9,
                        'flops_method': 'thop',
                        'mac': flops // 2,
                        'mac_G': flops / 2e9
                    })
                    print(f"✅ thop计算成功: {flops/1e9:.2f}G FLOPs")
                    return flops_info
                        
                 
                        
                except Exception as e:
                    print(f"❌ thop计算失败: {e}")
                    # 继续到估算方法
            
            # # 回退到估算方法
            # print("⚠️ 使用估算方法计算FLOPs")
            # flops = self._estimate_flops()
            # flops_info.update({
            #     'flops': flops,
            #     'flops_G': flops / 1e9,
            #     'flops_method': '估算',
            #     'mac': flops // 2,
            #     'mac_G': flops / 2e9
            # })
            # print(f"✅ 估算完成: {flops/1e9:.2f}G FLOPs")
            
        except Exception as e:
            print(f"❌ FLOPs计算完全失败: {e}")
            # 最后的回退
            try:
                flops = self._estimate_flops()
                flops_info.update({
                    'flops': flops,
                    'flops_G': flops / 1e9,
                    'flops_method': '估算(错误回退)',
                    'mac': flops // 2,
                    'mac_G': flops / 2e9
                })
                print(f"✅ 错误回退估算完成: {flops/1e9:.2f}G FLOPs")
            except Exception as final_e:
                print(f"❌ 连估算都失败了: {final_e}")
                # 设置默认值
                flops_info.update({
                    'flops': 0,
                    'flops_G': 0.0,
                    'flops_method': '计算失败',
                    'mac': 0,
                    'mac_G': 0.0
                })
        finally:
            # 恢复原始状态
            try:
                self.model.train(original_training)
                self.model.to(original_device)
            except:
                pass  # 确保即使恢复状态失败也不会中断程序
        
        return flops_info
    
    def _estimate_flops(self) -> float:
        """改进的FLOPs估算方法"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            h, w = self.input_size[1], self.input_size[2]
            
            # 基于参数量和输入尺寸的估算
            base_flops = total_params * 2  # 每个参数约2次乘加操作
            scale_factor = (h * w) / (800 * 1333)  # 相对于默认尺寸的缩放
            
            # 考虑不同类型层的复杂度
            conv_params = 0
            linear_params = 0
            
            for module in self.model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    conv_params += sum(p.numel() for p in module.parameters())
                elif isinstance(module, torch.nn.Linear):
                    linear_params += sum(p.numel() for p in module.parameters())
            
            # 卷积层的FLOPs计算更复杂
            conv_flops = conv_params * h * w * 2  # 卷积操作
            linear_flops = linear_params * 2  # 线性操作
            other_flops = (total_params - conv_params - linear_params) * 2
            
            estimated_flops = (conv_flops + linear_flops + other_flops) * 1.2  # 1.2是考虑其他操作的系数
            
            return estimated_flops
        except Exception as e:
            print(f"估算FLOPs也失败了: {e}")
            # 非常简单的估算
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                return total_params * 4  # 非常粗糙的估算
            except:
                return 0  # 实在不行就返回0
    
    def _estimate_memory(self) -> Dict[str, Any]:
        """估算内存占用"""
        try:
            # 参数内存
            param_memory = 0
            for param in self.model.parameters():
                param_memory += param.numel() * param.element_size()
                
            # 缓冲区内存
            buffer_memory = 0
            for buffer in self.model.buffers():
                buffer_memory += buffer.numel() * buffer.element_size()
            
            # 估算激活内存（基于输入尺寸）
            batch_size = 1
            input_memory = batch_size * self.input_size[0] * self.input_size[1] * self.input_size[2] * 4  # float32
            
            # 估算中间激活内存（改进的经验公式）
            total_params = sum(p.numel() for p in self.model.parameters())
            
            # 基于模型大小的激活内存估算
            if total_params < 1e6:  # 小模型
                activation_multiplier = 10
            elif total_params < 50e6:  # 中等模型
                activation_multiplier = 15
            else:  # 大模型
                activation_multiplier = 25
                
            activation_memory = input_memory * activation_multiplier
            
            total_memory = param_memory + buffer_memory + activation_memory
            
            return {
                'param_memory_MB': param_memory / (1024**2),
                'buffer_memory_MB': buffer_memory / (1024**2),
                'activation_memory_MB': activation_memory / (1024**2),
                'total_memory_MB': total_memory / (1024**2),
                'input_memory_MB': input_memory / (1024**2)
            }
        except Exception as e:
            print(f"内存估算失败: {e}")
            return {
                'param_memory_MB': 0.0,
                'buffer_memory_MB': 0.0,
                'activation_memory_MB': 0.0,
                'total_memory_MB': 0.0,
                'input_memory_MB': 0.0
            }
    
    def _measure_inference_speed(self, num_warmup=5, num_runs=20) -> Dict[str, Any]:
        """测量推理速度 - 减少测试次数防止卡死"""
        original_training = self.model.training
        original_device = next(self.model.parameters()).device
        
        try:
            self.model.eval()
            
            # 获取模型实际所在设备
            model_device = next(self.model.parameters()).device
            dummy_input = torch.randn(1, *self.input_size).to(model_device)
            
            # 检查模型是否能正常推理
            try:
                with torch.no_grad():
                    test_output = self.model(dummy_input)
                print(f"✅ 模型推理测试成功，设备: {model_device}")
            except Exception as e:
                print(f"❌ 模型推理测试失败: {e}")
                return {
                    'inference_time_ms': 0.0, 
                    'fps': 0.0, 
                    'throughput': 0.0,
                    'error': str(e)
                }
            
            # 预热 - 减少预热次数
            print(f"开始预热 {num_warmup} 次...")
            with torch.no_grad():
                for i in range(num_warmup):
                    _ = self.model(dummy_input)
                    if i == 0:
                        print("✅ 预热第一次成功")
            
            # 测速 - 减少测试次数
            print(f"开始性能测试 {num_runs} 次...")
            latencies = []
            
            if torch.cuda.is_available() and model_device.type == 'cuda':
                torch.cuda.synchronize(model_device)
            
            for i in range(num_runs):
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                if torch.cuda.is_available() and model_device.type == 'cuda':
                    torch.cuda.synchronize(model_device)
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                if i == 0:
                    print(f"✅ 第一次测试延迟: {latency_ms:.2f} ms")
            
            # 统计延迟
            avg_latency = sum(latencies) / len(latencies)
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
            
            print(f"✅ 性能测试完成: 平均延迟 {avg_latency:.2f} ms, FPS {fps:.1f}")
            
            return {
                'inference_time_ms': avg_latency,
                'fps': fps,
                'throughput': fps,
                'device': str(model_device)
            }
            
        except Exception as e:
            print(f"推理速度测试失败: {e}")
            return {
                'inference_time_ms': 0.0,
                'fps': 0.0,
                'throughput': 0.0,
                'error': str(e)
            }
        finally:
            try:
                self.model.train(original_training)
                self.model.to(original_device)
            except:
                pass  # 确保恢复状态不会中断程序
    
    def format_model_info(self) -> str:
        """格式化模型信息为可读字符串"""
        try:
            info = self.get_model_info()
            
            lines = [
                "=" * 80,
                "模型复杂度分析报告",
                "=" * 80,
                f"参数统计:",
                f"  总参数量: {info['params']:,} ({info['params_M']:.2f}M)",
                f"  可训练参数: {info['trainable_params']:,} ({info['trainable_params_M']:.2f}M)",
                f"  冻结参数: {info['frozen_params']:,} ({info['frozen_params_M']:.2f}M)",
                f"",
                f"计算复杂度:",
                f"  FLOPs: {info['flops']:,.0f} ({info['flops_G']:.2f}G)",
                f"  MACs: {info['mac']:,.0f} ({info['mac_G']:.2f}G)",
                f"  计算方法: {info['flops_method']}",
                f"",
                f"内存占用 :",
                f"  参数内存: {info['param_memory_MB']:.1f} MB",
                f"  缓冲区内存: {info['buffer_memory_MB']:.1f} MB",
                f"  激活内存: {info['activation_memory_MB']:.1f} MB",
                f"  总内存: {info['total_memory_MB']:.1f} MB",
                f"",
                # f"推理性能:",
                # f"  平均延迟: {info['inference_time_ms']:.1f} ms",
                # f"  FPS: {info['fps']:.1f}",
                # f"  吞吐量: {info['throughput']:.1f} images/sec",
            ]
            
            # 添加设备信息
            if 'device' in info:
                lines.append(f"  测试设备: {info['device']}")
            
            lines.append("=" * 80)
            
            return "\n".join(lines)
        except Exception as e:
            return f"模型信息格式化失败: {e}"


def calculate_model_stats(model, input_size=(3, 800, 1333), device='cuda'):
    """
    计算模型的参数量和FLOPs - 保持向后兼容
    """
    try:
        print("开始计算模型统计信息...")
        analyzer = ModelAnalyzer(model, input_size, device)
        info = analyzer.get_model_info()
        
        # 转换为原格式以保持兼容性
        stats = {
            'total_params': info['params'],
            'trainable_params': info['trainable_params'],
            'total_params_M': info['params_M'],
            'trainable_params_M': info['trainable_params_M'],
            'flops': info['flops'],
            'flops_G': info['flops_G'],
            'flops_method': info['flops_method']
        }
        
        print("✅ 模型统计信息计算完成")
        return stats
    except Exception as e:
        print(f"❌ 模型统计信息计算失败: {e}")
        # 返回默认值确保程序继续运行
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'total_params_M': total_params / 1e6,
                'trainable_params_M': trainable_params / 1e6,
                'flops': 0,
                'flops_G': 0.0,
                'flops_method': '计算失败'
            }
        except:
            return {
                'total_params': 0,
                'trainable_params': 0,
                'total_params_M': 0.0,
                'trainable_params_M': 0.0,
                'flops': 0,
                'flops_G': 0.0,
                'flops_method': '完全失败'
            }


def estimate_flops_roughly(model, input_size):
    """
    粗略估算模型的FLOPs - 保持向后兼容
    """
    try:
        analyzer = ModelAnalyzer(model, input_size)
        return analyzer._estimate_flops()
    except:
        return 0


def log_model_complexity(logger, model_stats):
    """记录模型复杂度信息 - 保持向后兼容"""
    try:
        logger.info("="*60)
        logger.info("模型复杂度统计")
        logger.info("="*60)
        logger.info(f"总参数量: {model_stats['total_params']:,} ({model_stats['total_params_M']:.2f}M)")
        logger.info(f"可训练参数量: {model_stats['trainable_params']:,} ({model_stats['trainable_params_M']:.2f}M)")
        
        if model_stats['flops'] > 0:
            logger.info(f"FLOPs: {model_stats['flops']:,} ({model_stats['flops_G']:.2f}G)")
            logger.info(f"FLOPs计算方法: {model_stats['flops_method']}")
        else:
            logger.info("FLOPs: 计算失败或未计算")
        
        logger.info("="*60)
    except Exception as e:
        logger.warning(f"记录模型复杂度失败: {e}")