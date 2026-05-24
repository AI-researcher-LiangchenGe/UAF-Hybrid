# UAF & HHAM-DRG: 两套高效长文本架构 / Two Efficient Long-Context Architectures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本仓库包含我在研究 Transformer 长文本瓶颈时提出的两个独立架构。  
This repository contains two independent architectures I proposed while researching Transformer long-context bottlenecks.

---

## 📄 论文 1: UAF (非归一化注意力流) / Paper 1: UAF (Unnormalized Attention Flow)

- **核心思想 / Core Idea**: 证明 Softmax 归一化必然导致注意力稀释，改用常微分方程状态积分实现 O(N) 线性复杂度。  
  Prove that Softmax normalization inevitably leads to attention dilution, and replace it with ODE state integration to achieve O(N) linear complexity.
- **论文 / Paper**: [paper_uaf.pdf](./paper_uaf.pdf)
- **代码入口 / Code Entry**: `uaf_hham_final.py` (使用 `--mode uaf` 运行 / run with `--mode uaf`)

## 📄 论文 2: HHAM-DRG (分层混合注意力记忆与动态残差门控) / Paper 2: HHAM-DRG (Hierarchical Hybrid Attentive Memory with Dynamic Residual Gating)

- **核心思想 / Core Idea**: 通过三级注意力、金字塔 KV 压缩和动态残差门控实现 91% 缓存压缩与 10 倍解码加速。  
  Achieve 91% KV cache compression and 10× decoding speedup via tri-level attention, pyramid KV management, and dynamic residual gating.
- **论文 / Paper**: [paper_hham.pdf](./paper_hham.pdf)
- **代码入口 / Code Entry**: `uaf_hham_final.py` (使用 `--mode hham` 运行 / run with `--mode hham`)

---

## 🔧 融合模式 / Hybrid Mode

两套架构可协同工作，在 UAF 全局建模后叠加 HHAM 局部精炼。  
The two architectures can work together: UAF provides global O(N) modeling, followed by HHAM local refinement.

运行命令 / Run command:
  python uaf_hham_final.py --mode hybrid

---

## 🚀 快速开始 / Quick Start

### 环境要求 / Requirements
- Python 3.8+
- PyTorch 2.0+
- safetensors (可选 / optional)

安装 / Install:
  pip install torch safetensors

### 运行演示 / Run Demo

默认融合模式 / Default hybrid mode:
  python uaf_hham_final.py

纯 UAF 模式 / Pure UAF mode:
  python uaf_hham_final.py --mode uaf

纯 HHAM 模式 / Pure HHAM mode:
  python uaf_hham_final.py --mode hham

---

## 📁 项目结构 / Project Structure

  .
  ├── README.md                   # 本文件 / This file
  ├── paper_uaf.pdf               # UAF 论文 / UAF paper
  ├── paper_hham.pdf              # HHAM-DRG 论文 / HHAM-DRG paper
  ├── uaf_hham_final.py           # 融合模型完整代码 / Full hybrid model code
  ├── code/                       # (可选) 独立模块 / (optional) standalone modules
  │   ├── uaf.py                  # UAF 独立实现 / Standalone UAF
  │   └── hham.py                 # HHAM 独立实现 / Standalone HHAM
  └── results/                    # 实验结果 / Experiment results

---

## 🧪 功能特性 / Features

- **O(N) 线性复杂度**：彻底抛弃 Softmax，使用状态积分。  
  **O(N) linear complexity**: Completely abandons Softmax in favor of state integration.
- **无 KV Cache 推理**：单步推理仅需一个固定大小的状态向量（KB 级）。  
  **KV-Cache-free inference**: Single-step inference needs only a fixed-size state vector (KB level).
- **三级注意力与动态门控**：在精度与效率间取得平衡。  
  **Tri-level attention & dynamic gating**: Balances accuracy and efficiency.
- **一键切换模式**：通过 `--mode` 参数可分别验证两篇论文。  
  **One-click mode switch**: Use `--mode` to validate each paper independently.
- **QLoRA 蒸馏就绪**：可无缝对接 `ms-swift` 等框架进行 4-bit 量化训练。  
  **QLoRA distillation ready**: Seamlessly integrates with frameworks like `ms-swift` for 4-bit quantized training.

---

## 📊 理论性能指标 / Theoretical Performance (100 万 Token / 1M Tokens, H100)

| 指标 / Metric | 原生 Transformer | UAF + HHAM-DRG |
| :--- | :--- | :--- |
| KV 缓存大小 / KV Cache Size | ~3,906 GB | ≤ 390 GB (压缩率 90%+ / >90% compression) |
| 单步解码延迟 / Per-step Latency | ~908 ms | ~91 ms (10× 加速 / 10× speedup) |
| 注意力复杂度 / Attention Complexity | O(N²) | O(N) |

---

## 📝 引用 / Citation

如果您使用了本工作，请引用对应的论文。  
If you use this work, please cite the corresponding paper.

### UAF
  @article{uaf2024,
    title={基于常微分方程的非归一化注意力流：一种替代Transformer的新架构},
    author={Your Name},
    year={2024}
  }

### HHAM-DRG
  @article{hham2024,
    title={分层混合注意力记忆与动态残差门控架构},
    author={Your Name},
    year={2024}
  }

---

## ⭐ 支持项目 / Support

如果这个项目对你有帮助，请给一个 Star ⭐  
If this project helps you, please give it a Star ⭐

---

**作者 / Author**: Liangchen Ge
**联系 / Contact**: 3974816442@qq.com
**许可证 / License**: MIT