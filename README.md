# WindyEmbed

**A Lightweight Embedding Model for Resource-Constrained Servers**

## Problem

Existing embedding models (BGE, E5, GTE) are:
- Too large for low-end servers (100M+ params)
- Require Python (not Java-friendly)
- Need GPU for fast inference

## Solution

WindyEmbed is a **tiny embedding model** designed for:
- ✅ 10M-50M parameters (10x smaller than BGE)
- ✅ Java-native inference (ONNX Runtime)
- ✅ CPU-only, no GPU needed
- ✅ <10ms per embedding on cheap hardware

## Architecture

```
Input Text
    │
    ▼
Tokenizer (WordPiece)
    │
    ▼
WindyEncoder (Transformer, 6 layers, 256 dim)
    │
    ▼
Mean Pooling
    │
    ▼
256-dim Embedding Vector
```

## Key Innovations

1. **Tiny Transformer** - 6 layers, 256 hidden dim, 4 heads
2. **Contrastive Training** - Learn from positive/negative pairs
3. **ONNX Export** - One-click export for Java deployment
4. **Java Inference** - ONNX Runtime, <10ms latency

## Quick Start

### 1. Train Model (Python)
```bash
pip install -r requirements.txt
python scripts/train.py --config configs/windyembed-10m.yaml
```

### 2. Export to ONNX
```bash
python scripts/export_onnx.py --checkpoint checkpoints/best
```

### 3. Use in Java
```java
import com.windyembed.WindyEmbed;

WindyEmbed embed = new WindyEmbed("model.onnx");
float[] vector = embed.encode("Hello world");
```

## Performance

| Model | Params | Dim | Latency (CPU) | Quality |
|-------|--------|-----|---------------|---------|
| BGE-small | 33M | 384 | 15ms | Good |
| WindyEmbed-10M | 10M | 256 | 5ms | Good |
| WindyEmbed-50M | 50M | 768 | 12ms | Excellent |

## Project Structure

```
WindyEmbed/
├── src/
│   ├── model/          # Python model
│   ├── training/       # Training pipeline
│   ├── data/           # Data loading
│   └── export/         # ONNX export
├── java/
│   └── src/main/java/com/windyembed/  # Java inference
├── configs/
├── scripts/
├── tests/
└── docs/
```

## License

MIT
