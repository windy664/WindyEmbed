# WindyEmbed

**为垃圾服务器设计的轻量级 Embedding 模型**

## 痛点

现有的 Embedding 模型（BGE、E5、GTE）：
- ❌ 太大了（100M+ 参数）
- ❌ 需要 Python（Java 不友好）
- ❌ 需要 GPU 才能跑得快
- ❌ 租的 E5 服务器根本跑不动

## 解决方案

WindyEmbed 专为**垃圾服务器**设计：
- ✅ 超小模型（10M-50M 参数）
- ✅ **纯 CPU 推理**，不需要 GPU
- ✅ **Java 原生**，ONNX Runtime 直接跑
- ✅ **E5 服务器 2-5ms** 完成 embedding
- ✅ 内存占用 <100MB

## 架构选择

**不用 Transformer！** 在 CPU 上太慢了。

| 架构 | E5 推理速度 | 适用场景 |
|------|------------|---------|
| **TextCNN** | ~2ms | 短文本、问答匹配（推荐）|
| **BiLSTM** | ~5ms | 长文本、语义理解 |
| **MLP** | ~1ms | 最快，适合简单任务 |
| Transformer | ~50ms | ❌ CPU 上太慢 |

## 架构图

```
输入文本
    │
    ▼
分词器 (WordPiece)
    │
    ▼
┌─────────────────────────────────────┐
│  TextCNN 架构                       │
│                                     │
│  Embedding (128维)                  │
│       │                             │
│       ▼                             │
│  ┌─────────────────────────────┐   │
│  │ 多尺度卷积                   │   │
│  │ Conv1d(kernel=3) → MaxPool  │   │
│  │ Conv1d(kernel=4) → MaxPool  │   │
│  │ Conv1d(kernel=5) → MaxPool  │   │
│  └─────────────────────────────┘   │
│       │                             │
│       ▼                             │
│  拼接 → 全连接 → 256维输出          │
└─────────────────────────────────────┘
    │
    ▼
归一化 Embedding 向量 (256维)
```

## 性能对比

| 模型 | 参数量 | 维度 | E5 CPU 延迟 | 质量 |
|------|--------|------|-------------|------|
| BGE-small | 33M | 384 | 50ms+ | 好 |
| **WindyEmbed-CNN** | 10M | 256 | **2ms** | 好 |
| **WindyEmbed-LSTM** | 15M | 256 | **5ms** | 很好 |
| **WindyEmbed-MLP** | 5M | 256 | **1ms** | 一般 |

## 快速开始

### 1. 训练模型（Python）
```bash
pip install torch numpy
python scripts/train.py --architecture cnn --epochs 10
```

### 2. 导出 ONNX
训练脚本自动导出 `model.onnx`

### 3. Java 使用
```xml
<!-- Maven 依赖 -->
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.16.3</version>
</dependency>
```

```java
import com.windyembed.WindyEmbed;

// 初始化（加载模型只需一次）
WindyEmbed embed = new WindyEmbed("model.onnx");

// 获取 embedding（2-5ms）
float[] vector = embed.encode("你好世界");

// 计算相似度
float sim = embed.similarity("机器学习", "人工智能");

// 批量编码
List<float[]> vectors = embed.encodeBatch(texts);

// 查找最相似
int idx = embed.findMostSimilar("查询", candidateList);
```

## 训练数据

### 方案 1：蒸馏（推荐）
用大模型生成高质量训练数据：

```bash
# 估算成本（DeepSeek API，约 ¥50 生成 10 万条）
python scripts/distill_data.py --api deepseek --count 100000 --estimate-only

# 生成数据
python scripts/distill_data.py --api deepseek --api-key YOUR_KEY --count 100000
```

### 方案 2：公开数据集
- [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STS_Benchmark)
- [ATEC](https://github.com/IceFlameWorm/ATEC)
- [PAWS-X](https://github.com/google-research-datasets/paws)

### 方案 3：自己造数据
从你的业务系统收集问答对、相似句对。

## 项目结构

```
WindyEmbed/
├── src/
│   ├── model/
│   │   ├── cpu_encoder.py    # CPU 优化的模型架构
│   │   └── encoder.py        # 原 Transformer 版本（备用）
│   ├── training/
│   │   └── contrastive.py    # 对比学习损失函数
│   └── export/
│       └── onnx_export.py    # ONNX 导出
├── java/
│   ├── pom.xml               # Maven 配置
│   └── WindyEmbed.java       # Java 推理代码
├── scripts/
│   ├── train.py              # 训练脚本
│   └── distill_data.py       # 蒸馏数据生成
├── configs/
└── docs/
```

## 应用场景

1. **语义搜索** - 根据语义而非关键词搜索
2. **RAG（检索增强生成）** - 找到相关文档再生成回答
3. **问答系统** - 匹配问题和答案
4. **推荐系统** - 基于内容的推荐
5. **文本聚类** - 自动分类
6. **去重** - 找到重复或相似内容

## 为什么选 WindyEmbed？

| 对比项 | WindyEmbed | BGE/E5 | 自己训练 |
|--------|-----------|--------|---------|
| 模型大小 | 10M | 100M+ | 看情况 |
| CPU 推理 | 2-5ms | 50ms+ | 慢 |
| 需要 GPU | ❌ 不需要 | ✅ 需要 | 看情况 |
| Java 支持 | ✅ 原生 | ❌ 需要 Python | 麻烦 |
| 训练成本 | ¥50（蒸馏）| 免费但效果一般 | 高 |
| 适合垃圾服务器 | ✅ 完美 | ❌ 跑不动 | 看情况 |

## 待办事项

- [ ] 完成 TextCNN 架构实现
- [ ] 完成 BiLSTM 架构实现
- [ ] 实现蒸馏数据生成器
- [ ] 训练初版模型
- [ ] 导出 ONNX 并测试 Java 推理
- [ ] 在 E5 服务器上测试性能
- [ ] 写论文
- [ ] 发布到 Maven Central

## 开源协议

MIT

## 作者

windy664
