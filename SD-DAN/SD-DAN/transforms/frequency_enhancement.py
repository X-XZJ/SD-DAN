import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


class HighLowFrequencyMixedEnhancement:
    """
    High-Low Frequency Mixed Enhancement
    利用傅里叶变换进行频域增强,通过高斯高通和低通滤波器混合不同频率分量
    """
    
    def __init__(self, sigma=30, alpha=0.3, beta=0.3, gamma=0.4, apply_prob=1.0):
        """
        Args:
            sigma: 高斯滤波器的标准差,控制滤波强度
            alpha: 高频分量的权重
            beta: 低频分量的权重  
            gamma: 原始频域的权重
            apply_prob: 应用此增强的概率
        """
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "alpha + beta + gamma must equal 1.0"
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.apply_prob = apply_prob
    
    def __call__(self, img, target=None):
        """
        Args:
            img: PIL Image or torch.Tensor
            target: 目标标注(可选)
        Returns:
            增强后的图像和目标
        """
        if np.random.random() > self.apply_prob:
            return img if target is None else (img, target)
        
        # 转换为numpy数组
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            is_pil = True
        elif isinstance(img, torch.Tensor):
            img_array = img.permute(1, 2, 0).cpu().numpy()
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            is_pil = False
        else:
            img_array = img
            is_pil = False
        
        # 处理RGB图像的每个通道
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            enhanced_channels = []
            for c in range(3):
                enhanced_channel = self._process_channel(img_array[:, :, c])
                enhanced_channels.append(enhanced_channel)
            enhanced_img = np.stack(enhanced_channels, axis=2)
        else:
            # 单通道图像
            enhanced_img = self._process_channel(img_array)
        
        # 转换回原始格式
        if is_pil:
            enhanced_img = Image.fromarray(enhanced_img.astype(np.uint8))
        elif isinstance(img, torch.Tensor):
            enhanced_img = torch.from_numpy(enhanced_img).permute(2, 0, 1).float() / 255.0
            if img.device != torch.device('cpu'):
                enhanced_img = enhanced_img.to(img.device)
        
        return enhanced_img if target is None else (enhanced_img, target)
    
    def _process_channel(self, channel):
        """
        对单个通道进行频域增强
        Args:
            channel: H x W 的单通道图像
        Returns:
            增强后的单通道图像
        """
        H, W = channel.shape
        
        # 转换为float32
        I = channel.astype(np.float32)
        
        # 1. 傅里叶变换: 将图像从空间域转换到频域
        F = np.fft.fft2(I)
        F_shifted = np.fft.fftshift(F)  # 将零频分量移到中心
        
        # 2. 创建高斯高通滤波器和低通滤波器
        # 创建频率网格
        u = np.arange(H)
        v = np.arange(W)
        u, v = np.meshgrid(u - H // 2, v - W // 2, indexing='ij')
        
        # 计算到中心的距离 D(u,v)
        D = np.sqrt(u**2 + v**2)
        
        # 高斯低通滤波器: H_lp(u,v) = exp(-D(u,v)^2 / (2*sigma^2))
        H_lp = np.exp(-(D**2) / (2 * self.sigma**2))
        
        # 高斯高通滤波器: H_hp(u,v) = 1 - exp(-D(u,v)^2 / (2*sigma^2))
        H_hp = 1 - H_lp
        
        # 3. 应用滤波器得到高频和低频分量
        F_high = H_hp * F_shifted  # 高频分量
        F_low = H_lp * F_shifted   # 低频分量
        
        # 4. 自适应混合不同频率分量
        # F_mix(u,v) = alpha * F_high(u,v) + beta * F_low(u,v) + gamma * F(u,v)
        F_mix = self.alpha * F_high + self.beta * F_low + self.gamma * F_shifted
        
        # 5. 逆傅里叶变换: 将频域信息转换回空间域
        F_mix_shifted = np.fft.ifftshift(F_mix)
        I_mix = np.fft.ifft2(F_mix_shifted)
        I_mix = np.real(I_mix)  # 取实部
        
        # 6. 归一化到 [0, 255]
        I_mix = np.clip(I_mix, 0, 255)
        
        return I_mix
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"sigma={self.sigma}, "
                f"alpha={self.alpha}, "
                f"beta={self.beta}, "
                f"gamma={self.gamma}, "
                f"apply_prob={self.apply_prob})")


class AdaptiveFrequencyEnhancement(HighLowFrequencyMixedEnhancement):
    """
    自适应频率增强版本,可以根据图像内容动态调整参数
    """
    
    def __init__(self, sigma_range=(20, 40), alpha_range=(0.2, 0.4), 
                 beta_range=(0.2, 0.4), apply_prob=1.0):
        """
        Args:
            sigma_range: sigma的范围,用于随机采样
            alpha_range: alpha的范围
            beta_range: beta的范围
            apply_prob: 应用此增强的概率
        """
        self.sigma_range = sigma_range
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.apply_prob = apply_prob
    
    def __call__(self, img, target=None):
        if np.random.random() > self.apply_prob:
            return img if target is None else (img, target)
        
        # 随机采样参数
        sigma = np.random.uniform(*self.sigma_range)
        alpha = np.random.uniform(*self.alpha_range)
        beta = np.random.uniform(*self.beta_range)
        gamma = 1.0 - alpha - beta
        
        # 临时设置参数
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        return super().__call__(img, target)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"sigma_range={self.sigma_range}, "
                f"alpha_range={self.alpha_range}, "
                f"beta_range={self.beta_range}, "
                f"apply_prob={self.apply_prob})")