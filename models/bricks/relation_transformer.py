import copy
import functools
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.bricks.misc import Conv2dNormActivation
from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import get_sine_pos_embed
from util.misc import inverse_sigmoid

INIT_THRESHOLD = 0.5  # threshold 默认初始值
FEATURE_ENHANCEMENT_CHANNELS = 256  # 特征增强的通道数,默认与 embed_dim 一致


class BackgroundSmoothingModule(nn.Module):
    """Background smoothing with channel and spatial branches"""
    def __init__(self, channels, reduction=4, spatial_scale=2, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.spatial_scale = spatial_scale
        
        # Channel branch
        padding = (kernel_size - 1) // 2
        self.channel_compress = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.channel_expand = nn.Sequential(
            nn.Conv2d(channels // reduction, channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        
        # Learnable fusion weights
        self.w_alpha = nn.Parameter(torch.zeros(1))
        self.w_beta = nn.Parameter(torch.zeros(1))
        self.w_gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, background_features):
        """
        Args:
            background_features: B1, shape [N, C, H, W]
        Returns:
            MB1: enhanced background features
        """
        N, C, H, W = background_features.shape
        
        # Channel branch: MB1_channel
        compressed = self.channel_compress(background_features)
        expanded = self.channel_expand(compressed)
        MB1_channel = expanded + background_features
        
        # Spatial branch: MB1_spatial
        B1_down = F.avg_pool2d(background_features, kernel_size=self.spatial_scale, 
                                stride=self.spatial_scale)
        # 使用size参数而不是scale_factor,确保输出尺寸与输入完全一致
        MB1_spatial = F.interpolate(B1_down, size=(H, W), 
                                     mode='bilinear', align_corners=False)
        
        # Adaptive fusion - 修正 softmax 调用
        weights = torch.stack([self.w_alpha, self.w_beta, self.w_gamma])
        weights = torch.softmax(weights, dim=0)  # 修改这里
        alpha, beta, gamma = weights[0], weights[1], weights[2]
        
        MB1_enhanced = alpha * MB1_channel + beta * MB1_spatial + gamma * background_features
        
        return MB1_enhanced


class FrequencyEnhancement(nn.Module):
    """Frequency domain enhancement using FFT"""
    def __init__(self, sigma=10.0):
        super().__init__()
        self.sigma = sigma
        
    def gaussian_lowpass_filter(self, shape, sigma, device):
        """Create Gaussian low-pass filter"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        y = torch.arange(0, rows, device=device).view(-1, 1) - crow
        x = torch.arange(0, cols, device=device).view(1, -1) - ccol
        
        mask = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return mask
    
    def gaussian_highpass_filter(self, shape, sigma, device):
        """Create Gaussian high-pass filter"""
        return 1 - self.gaussian_lowpass_filter(shape, sigma, device)
    
    def lowpass_filter(self, features):
        """Apply low-pass filter (for background)"""
        N, C, H, W = features.shape
        result = torch.zeros_like(features)
        
        for n in range(N):
            for c in range(C):
                # FFT
                fft = torch.fft.fft2(features[n, c])
                fft_shift = torch.fft.fftshift(fft)
                
                # Apply low-pass filter
                mask = self.gaussian_lowpass_filter((H, W), self.sigma, features.device)
                fft_filtered = fft_shift * mask
                
                # IFFT
                fft_ishift = torch.fft.ifftshift(fft_filtered)
                result[n, c] = torch.real(torch.fft.ifft2(fft_ishift))
        
        return result
    
    def highpass_filter(self, features):
        """Apply high-pass filter (for foreground)"""
        N, C, H, W = features.shape
        result = torch.zeros_like(features)
        
        for n in range(N):
            for c in range(C):
                # FFT
                fft = torch.fft.fft2(features[n, c])
                fft_shift = torch.fft.fftshift(fft)
                
                # Apply high-pass filter
                mask = self.gaussian_highpass_filter((H, W), self.sigma, features.device)
                fft_filtered = fft_shift * mask
                
                # IFFT
                fft_ishift = torch.fft.ifftshift(fft_filtered)
                result[n, c] = torch.real(torch.fft.ifft2(fft_ishift))
        
        return result


class LearnableMaskGenerator(nn.Module):
    def __init__(self, kernel_size=7, init_threshold=0.5, temperature=10.0):
        super().__init__()
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                                      padding=kernel_size//2, bias=False)
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        self.temperature = temperature
        print(f"✓ LearnableMaskGenerator 已初始化,init_threshold={init_threshold}, temperature={temperature}")
        
    def forward(self, features):
        N, C, H, W = features.shape
        
        wavg = torch.mean(features, dim=1, keepdim=True)
        wmax, _ = torch.max(features, dim=1, keepdim=True)
        concat_features = torch.cat([wavg, wmax], dim=1)
        Ms = torch.sigmoid(self.spatial_conv(concat_features))
        
        # 使用可微分的软阈值
        fg_mask = torch.sigmoid((Ms - self.threshold) * self.temperature)
        bg_mask = 1.0 - fg_mask
        
        return fg_mask, bg_mask, Ms


class ForegroundChannelEnhancement(nn.Module):
    """前景通道增强模块: 先升维再降维"""
    def __init__(self, channels, expansion=2, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        
        padding = (kernel_size - 1) // 2
        # 通道扩展
        self.channel_expand = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        # 通道压缩
        self.channel_compress = nn.Sequential(
            nn.Conv2d(channels * expansion, channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, foreground_features):
        """
        Args:
            foreground_features: F, shape [N, C, H, W]
        Returns:
            MF_channel: 通道增强后的前景特征 [N, C, H, W]
        """
        expanded = self.channel_expand(foreground_features)
        compressed = self.channel_compress(expanded)
        MF_channel = compressed + foreground_features  # 残差连接
        
        return MF_channel


class ForegroundSpatialEnhancement(nn.Module):
    """前景空间增强模块: 上采样->卷积增强->下采样"""
    def __init__(self, channels, scale_factor=2, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        
        padding = (kernel_size - 1) // 2
        self.conv_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, foreground_features):
        """
        Args:
            foreground_features: F, shape [N, C, H, W]
        Returns:
            MF_spatial: 空间增强后的前景特征 [N, C, H, W]
        """
        N, C, H, W = foreground_features.shape
        
        # Step 1: 空间上采样
        F_up = F.interpolate(
            foreground_features,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )
        
        # Step 2: 局部增强 (在高分辨率空间)
        F_enh = self.conv_enhance(F_up)
        
        # Step 3: 下采样回到原尺度
        MF_spatial = F.avg_pool2d(
            F_enh,
            kernel_size=self.scale_factor,
            stride=self.scale_factor
        )
        
        # 确保输出尺寸与输入一致
        if MF_spatial.shape[2:] != (H, W):
            MF_spatial = F.interpolate(MF_spatial, size=(H, W), mode='bilinear', align_corners=False)
        
        return MF_spatial


class FeatureEnhancementModule(nn.Module):
    """Feature enhancement module for foreground and background"""
    def __init__(self, channels, reduction=4, spatial_scale=2, kernel_size=5, sigma=10.0):
        super().__init__()
        self.channels = channels
        
        # 使用全局变量作为初始阈值
        global INIT_THRESHOLD
        self.mask_generator = LearnableMaskGenerator(kernel_size=7, init_threshold=INIT_THRESHOLD, temperature=20)
        print(f"✓ LearnableMaskGenerator 已初始化,init_threshold={INIT_THRESHOLD}")
        
        # Background smoothing modules
        self.background_smooth = BackgroundSmoothingModule(channels, reduction, 
                                                           spatial_scale, kernel_size)
        self.freq_enhance = FrequencyEnhancement(sigma)
        
        # 新的前景增强模块
        # Step 8: 通道增强 (先升维再降维, r=2, K=3)
        self.foreground_channel_enhance = ForegroundChannelEnhancement(channels, expansion=2, kernel_size=3)
        
        # Step 9: 空间增强 (上采样->卷积->下采样, scale=2)
        self.foreground_spatial_enhance = ForegroundSpatialEnhancement(channels, scale_factor=2, kernel_size=3)
        
        # Step 9: 自适应融合权重 (α, β)
        self.w_alpha_fg = nn.Parameter(torch.zeros(1))
        self.w_beta_fg = nn.Parameter(torch.zeros(1))
    
    def forward(self, features, bboxes=None):
        """
        Args:
            features: A, backbone last layer features [N, C, H, W]
            bboxes: not used anymore, kept for API compatibility
        Returns:
            P: enhanced features [N, C, H, W]
        """
        N, C, H, W = features.shape
        
        # Generate learnable masks from features
        fg_mask, bg_mask, Ms = self.mask_generator(features)
        
        # Step 1: Extract foreground features using learnable mask
        F = features * fg_mask  # [N, C, H, W]
        
        # Step 2: Extract background features using learnable mask
        B1 = features * bg_mask  # [N, C, H, W]
        
        # Step 3-5: Background channel and spatial smoothing
        MB1 = self.background_smooth(B1)
        
        # Step 6: Background frequency smoothing
        MB2 = self.freq_enhance.lowpass_filter(B1)
        
        # Step 7: Combine background enhancements
        B = MB1 + MB2
        
        # 新的前景增强流程
        # Step 8: 前景通道增强 (先升维再降维)
        F4 = self.foreground_channel_enhance(F)  # MF_channel -> F4
        
        # Step 9: 前景空间增强 (上采样->卷积->下采样)
        F7 = self.foreground_spatial_enhance(F)  # MF_spatial -> F7
        
        # Step 9: 自适应融合 F7 和 F4 - 修正 softmax 调用
        weights = torch.stack([self.w_alpha_fg, self.w_beta_fg])
        weights = torch.softmax(weights, dim=0)  # 修改这里
        alpha_fg, beta_fg = weights[0], weights[1]
        F8 = alpha_fg * F7 + beta_fg * F4
        
        # Step 10: 前景频域增强
        F5 = self.freq_enhance.highpass_filter(F)
        F3 = F5 + F
        
        # Step 10: 组合前景增强
        F9 = F3 + F8
        
        # Final: Combine foreground and background
        P = F9 + B
        
        return P


class RelationTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        hybrid_num_proposals: int = 900,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.hybrid_num_proposals = hybrid_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.hybrid_tgt_embed = nn.Embedding(hybrid_num_proposals, self.embed_dim)
        self.hybrid_class_head = nn.Linear(self.embed_dim, num_classes)
        self.hybrid_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        nn.init.normal_(self.hybrid_tgt_embed.weight)
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        nn.init.constant_(self.hybrid_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        gt_bboxes=None,
        noised_label_query=None,
        noised_box_query=None,
        attn_mask=None,
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
            gt_bboxes=gt_bboxes,
            multi_level_feats=multi_level_feats,
        )

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        topk, num_classes = self.two_stage_num_proposals, self.num_classes
        topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        topk = self.hybrid_num_proposals if self.training else 0
        if self.training:
            # get hybrid classes and coordinates, target and reference points
            hybrid_enc_class = self.hybrid_class_head(output_memory)
            hybrid_enc_coord = self.hybrid_bbox_head(output_memory) + output_proposals
            hybrid_enc_coord = hybrid_enc_coord.sigmoid()
            topk_index = torch.topk(hybrid_enc_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
            hybrid_enc_class = hybrid_enc_class.gather(
                1, topk_index.expand(-1, -1, self.num_classes)
            )
            hybrid_enc_coord = hybrid_enc_coord.gather(1, topk_index.expand(-1, -1, 4))
            hybrid_reference_points = hybrid_enc_coord.detach()
            hybrid_target = self.hybrid_tgt_embed.weight.expand(
                multi_level_feats[0].shape[0], -1, -1
            )
        else:
            hybrid_enc_class = None
            hybrid_enc_coord = None

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        if self.training:
            hybrid_classes, hybrid_coords = self.decoder(
                query=hybrid_target,
                value=memory,
                key_padding_mask=mask_flatten,
                reference_points=hybrid_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                skip_relation=True,
            )
        else:
            hybrid_classes = hybrid_coords = None

        return (
            outputs_classes,
            outputs_coords,
            enc_outputs_class,
            enc_outputs_coord,
            hybrid_classes,
            hybrid_coords,
            hybrid_enc_class,
            hybrid_enc_coord,
        )


class RelationTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim
        self.memory_fusion = nn.Sequential(
            nn.Linear((num_layers + 1) * self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        
        # 在初始化时就创建 feature_enhancement
        global FEATURE_ENHANCEMENT_CHANNELS
        channels = FEATURE_ENHANCEMENT_CHANNELS if FEATURE_ENHANCEMENT_CHANNELS > 0 else self.embed_dim
        self.feature_enhancement = FeatureEnhancementModule(channels)
        print(f"✓ FeatureEnhancementModule 已初始化,channels={channels}")

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None,
        gt_bboxes=None,
        multi_level_feats=None,
    ):
        # Feature enhancement for the first encoder layer
        if multi_level_feats is not None:
            # Get the last level features (highest resolution backbone feature)
            last_feat = multi_level_feats[-1]  # [N, C, H, W]
            N, C, H, W = last_feat.shape
            
            # 直接使用已初始化的 feature_enhancement
            enhanced_feat = self.feature_enhancement(last_feat)  # [N, C, H, W]
            
            # Flatten enhanced feature
            enhanced_feat_flatten = enhanced_feat.flatten(2).transpose(1, 2)  # [N, H*W, C]
            
            # Replace the last level features in query
            last_level_idx = len(spatial_shapes) - 1
            if last_level_idx >= 0:
                start_idx = level_start_index[last_level_idx]
                end_idx = level_start_index[last_level_idx + 1] if last_level_idx + 1 < len(level_start_index) else query.shape[1]
                
                # Replace with enhanced features
                query = torch.cat([
                    query[:, :start_idx],
                    enhanced_feat_flatten,
                    query[:, end_idx:] if end_idx < query.shape[1] else torch.empty(0, device=query.device)
                ], dim=1)
        
        # Original encoder forward
        queries = [query]
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )
            queries.append(query)
        query = torch.cat(queries, -1)
        query = self.memory_fusion(query)
        return query


class RelationTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=query,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class RelationTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        # relation embedding
        self.position_relation_embedding = PositionRelationEmbedding(16, self.num_heads)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initialize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
        skip_relation=False,
    ):
        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        pos_relation = attn_mask  # fallback pos_relation to attn_mask
        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(
                reference_points_input[:, :, 0, :], self.embed_dim // 2
            )
            query_pos = self.ref_point_head(query_sine_embed)
            query_pos = query_pos * self.query_scale(query) if layer_idx != 0 else query_pos

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=pos_relation,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query))
            output_coord = output_coord + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # calculate position relation embedding
            if not skip_relation:
                src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
                tgt_boxes = output_coord
                pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
                if attn_mask is not None:
                    pos_relation.masked_fill_(attn_mask, float("-inf"))

            # iterative bounding box refinement
            reference_points = inverse_sigmoid(reference_points.detach())
            reference_points = self.bbox_head[layer_idx](query) + reference_points
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords


class RelationTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
            need_weights=False,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed


class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.0,
        scale=100.0,
        activation_layer=nn.ReLU,
        inplace=True,
    ):
        super().__init__()
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

    def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        # src_boxes: [batch_size, num_boxes1, 4]
        # tgt_boxes: [batch_size, num_boxes2, 4]
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
        with torch.no_grad():
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone()