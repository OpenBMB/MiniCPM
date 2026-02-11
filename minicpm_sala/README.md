# MiniCPM-SALA

[🤗 Huggingface](https://huggingface.co/openbmb/MiniCPM-SALA) | [ 📖Technical Blog](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf)

## What is MiniCPM-SALA?

**MiniCPM-SALA (Sparse Attention and Linear Attention)** introduces the first large-scale hybrid architecture that systematically integrates **25% sparse attention** (InfLLM-v2) with **75% linear attention** (Lightning Attention) for efficient ultra-long context modeling.

By combining high-fidelity long context modeling with globally efficient recurrent computation—and further empowered by **HyPE**, a hybrid positional embedding scheme—the model scales to million-token context windows while preserving strong length generalization.

- **Performance:** Compared to dense Transformer baselines (e.g., Qwen3-8B), MiniCPM-SALA achieves up to **3.5× inference speed** under long-context settings, significantly reducing both compute and KV-cache overhead.
- **Methodology:** To ensure performance retention, we propose a novel Transformer-to-hybrid distillation recipe, initializing from MiniCPM-4 and applying structured decay and post-training adaptation to effectively transfer dense attention capabilities into the hybrid architecture.

------

# MiniCPM-SALA Cookbook

Cook up amazing long-context applications efficiently with MiniCPM-SALA, bringing unparalleled context understanding and speed right to your fingertips!

## ✨ What Makes Our Recipes Special?

### Easy Usage Documentation

Our comprehensive documentation website presents every recipe in a clear, well-organized manner. All features are displayed at a glance, making it easy for you to quickly find exactly what you need.

### Broad User Spectrum

We support a wide range of users, from individuals to enterprises and researchers.

- **Individuals:** Enjoy effortless inference using HuggingFace with minimal setup.
- **Enterprises:** Achieve high-throughput, scalable performance with vLLM or SGLang.
- **Researchers:** Leverage advanced frameworks, including [Transformers Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) and [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), to enable flexible model development and cutting-edge experimentation.

### Fine-tuning recipes

Customize your model with your own ingredients. For more detailed instructions for fine-tuning, check out the `finetune` subdirectory and its corresponding [`README.md`](./finetune/README.md).

### Training

We provide training methods serving different needs as follows:

| **Framework**                                    | **Description**                            |
| ------------------------------------------------ | ------------------------------------------ |
| [**Transformers Trainer**](./finetune/README.md#hugging-face-transformers-trainer) | Most flexible for low-level customization. |
| [**LLaMA-Factory**](./finetune/README.md#llama-factory)        | Modular fine-tuning toolkit.               |

------

## 👥 Community

### Contributing

We love new recipes! Please share your creative dishes:

1. Fork the repository
2. Create your recipe
3. Submit a pull request

### Issues & Support

- Found a bug? [Open an issue](https://github.com/OpenBMB/MiniCPM/issues)

### Institutions

This cookbook is developed by [**OpenBMB**](https://github.com/OpenBMB).

------

## 📜 License

This cookbook is served under the [Apache-2.0 License](../LICENSE) - cook freely, share generously! 🍳

## Citation

If you find our model, code, or paper helpful, please consider citing our papers 📝 and starring us ⭐️!

```bibtex
@article{minicpm4,
  title={{MiniCPM-SALA}: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling},
  author={MiniCPM Team},
  year={2026}
}
```