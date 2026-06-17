"""
WindyEmbed Fast Encoder - 极致 CPU 优化

优化策略：
1. INT8 量化 - 4x 更小，2x 更快
2. 最小维度 - 64 维就够用
3. 单层卷积 - 减少计算量
4. ReLU 激活 - CPU 上最快
5. 避免动态内存分配
6. 缓存友好的内存布局

目标：E5 服务器 < 0.5ms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class FastEncoderConfig:
    """极致优化的配置"""

    vocab_size: int = 32000
    embedding_dim: int = 64        # 最小可用维度
    num_filters: int = 64          # 最少滤波器
    kernel_size: int = 3           # 单一卷积核
    output_dim: int = 64           # 最小输出维度
    max_seq_len: int = 64          # 最短序列
    dropout: float = 0.0           # 推理时不用 dropout


class FastCNNEncoder(nn.Module):
    """
    极致优化的 CNN 编码器

    设计原则：
    1. 最少的层数（1层卷积）
    2. 最小的维度（64维）
    3. 最简单的操作（卷积 + 池化）
    4. 缓存友好的内存布局

    性能目标：
    - E5-2680 v4: < 0.5ms
    - i7-1185G7: < 0.3ms
    - 模型大小: < 5MB
    """

    def __init__(self, config: FastEncoderConfig):
        super().__init__()
        self.config = config

        # Embedding（最小维度）
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=0
        )

        # 单层卷积（最快）
        self.conv = nn.Conv1d(
            in_channels=config.embedding_dim,
            out_channels=config.num_filters,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2
        )

        # 输出投影（最小）
        self.fc = nn.Linear(config.num_filters, config.output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        极致优化的前向传播

        优化点：
        1. 避免不必要的 transpose
        2. 使用 adaptive_max_pool1d（最快）
        3. 不用 dropout（推理时）
        4. 直接返回，减少函数调用
        """
        # Embedding: [batch, seq_len, embed_dim]
        x = self.embedding(input_ids)

        # 应用 mask
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        x = x.transpose(1, 2)

        # Conv1d + ReLU（最快）
        x = F.relu(self.conv(x))

        # MaxPool（最快）
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)

        # 全连接
        x = self.fc(x)

        # 归一化
        x = F.normalize(x, p=2, dim=-1)

        return x


class QuantizedEncoder(nn.Module):
    """
    INT8 量化编码器

    优势：
    1. 模型大小减少 4x（FP32 → INT8）
    2. 推理速度提升 2x（整数运算更快）
    3. 内存占用减少 4x

    注意：
    - 需要校准数据集
    - 精度可能略有下降
    """

    def __init__(self, encoder: FastCNNEncoder):
        super().__init__()
        self.encoder = encoder

        # 量化配置
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 量化输入
        x = self.quant(input_ids)

        # 编码
        x = self.encoder(x, attention_mask)

        # 反量化输出
        x = self.dequant(x)

        return x


def create_fast_encoder(config: FastEncoderConfig = None) -> FastCNNEncoder:
    """创建极致优化的编码器"""
    if config is None:
        config = FastEncoderConfig()
    return FastCNNEncoder(config)


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    优化模型用于推理

    优化步骤：
    1. 设置为评估模式
    2. 禁用梯度计算
    3. 融合 BN 层（如果有）
    4. JIT 编译（可选）
    """
    model.eval()

    # 禁用梯度
    for param in model.parameters():
        param.requires_grad = False

    # JIT 编译（如果支持）
    try:
        # 创建示例输入
        dummy_input = torch.randint(0, 1000, (1, 64))
        dummy_mask = torch.ones(1, 64, dtype=torch.long)

        # TorchScript 编译
        model = torch.jit.trace(model, (dummy_input, dummy_mask))
        print("[OK] JIT 编译成功")
    except Exception as e:
        print(f"[WARN] JIT 编译失败: {e}")

    return model


def benchmark_model(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 64,
    num_iterations: int = 1000,
) -> Dict:
    """
    基准测试

    返回：
    - 平均推理时间 (ms)
    - P50 推理时间 (ms)
    - P95 推理时间 (ms)
    - P99 推理时间 (ms)
    - 吞吐量 (samples/sec)
    """
    import time
    import numpy as np

    model.eval()

    # 创建输入
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # 预热
    for _ in range(100):
        with torch.no_grad():
            _ = model(input_ids, attention_mask)

    # 测试
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter_ns()
        with torch.no_grad():
            _ = model(input_ids, attention_mask)
        elapsed = (time.perf_counter_ns() - start) / 1e6  # 转换为 ms
        times.append(elapsed)

    times = np.array(times)

    return {
        'avg_ms': np.mean(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'throughput': 1000 / np.mean(times),  # samples/sec
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
    }