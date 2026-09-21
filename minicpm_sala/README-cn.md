# MiniCPM-SALA

[🤗 Huggingface](https://huggingface.co/openbmb/MiniCPM-SALA) | [📖 技术博客](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf)

# MiniCPM-SALA 是什么？

**MiniCPM-SALA（Sparse Attention and Linear Attention）** 引入了首个大规模混合架构，系统性地将 **25% 的稀疏注意力**（InfLLM-v2）与 **75% 的线性注意力**（Lightning Attention）相结合，用于高效的超长上下文建模。

通过将高保真度的局部建模与全局高效的循环计算相结合——并进一步得到 **HyPE**的赋能——该模型能够扩展到百万令牌级别的上下文窗口，同时保持强大的长度泛化能力。

- **性能：** 与 Transformer 基线模型（如 Qwen3-8B）相比，MiniCPM-SALA 在长上下文设置下实现了高达 **3.5 倍的推理加速**，显著降低了计算和 KV 缓存的开销。
- **方法：** 为确保性能保持，我们提出了一种新颖的 Transformer 到混合架构的蒸馏方案，从 MiniCPM-4 初始化，并应用结构化衰减和后训练适应，以有效地将密集注意力能力迁移到混合架构中。

------

# MiniCPM-SALA 使用指南

使用 MiniCPM-SALA 高效地构建惊人的长上下文应用，为您带来无与伦比的上下文理解能力和速度！

## ✨ 有何特别之处？

### 易于使用的文档

我们全面的文档网站以清晰、有条理的方式呈现每一条指南。所有功能一目了然，助您快速找到所需内容。

### 广泛的用户支持

我们支持从个人到企业和研究者的广泛用户群体。

- **个人用户：** 通过 HuggingFace 轻松进行推理，设置简单。
- **企业用户：** 借助 vLLM 或 SGLang 实现高吞吐、可扩展的性能。
- **研究者：** 利用 [Transformers Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)、[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) 等高级框架，进行灵活的模型开发和前沿实验。

### 微调指南

使用您自己的“食材”定制模型。更详细的微调说明，请查看 `finetune` 子目录及其对应的 [`README.md`](.finetune/README.md) 文件。

### 训练

我们提供满足不同需求的训练方法，如下所示：

| **框架**                                         | **描述**                       |
| ------------------------------------------------ | ------------------------------ |
| [**Transformers Trainer**](./finetune/README-cn.md#hugging-face-transformers-trainer) | 最适合底层自定义，灵活性最高。 |
| [**LLaMA-Factory**](./finetune/README-cn.md#llama-factory)        | 模块化微调工具包。             |

------

## 👥 社区

### 贡献

我们欢迎新的“食谱”！请分享您的创意作品：

1. Fork 本仓库
2. 创建您的“食谱”
3. 提交 Pull Request

### 问题与支持

- 发现了错误？[提交 Issue](https://github.com/OpenBMB/MiniCPM/issues)

### 机构

本指南由 [**OpenBMB**](https://github.com/OpenBMB) 开发。

------

## 📜 许可证

本指南基于 [Apache-2.0 License](../LICENSE) 提供——自由“烹饪”，慷慨分享！🍳

## 引用

如果您觉得我们的模型、代码或论文有所帮助，请考虑引用我们的论文 📝 并为我们点亮星标 ⭐️！

```bibtex
@article{minicpm4,
  title={{MiniCPM-SALA}: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling},
  author={MiniCPM Team},
  year={2026}
}
```