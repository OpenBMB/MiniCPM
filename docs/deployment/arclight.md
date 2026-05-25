# Deploy ArcLight from Source (CPU)

[ArcLight](https://github.com/OpenBMB/ArcLight) is a lightweight LLM inference framework written in C/C++ for unified-memory systems. It is designed for inference scenarios beyond high-performance GPU servers, with current v1.0 optimizations focused on many-core CPU platforms and cross-NUMA tensor parallelism.

ArcLight currently supports CPU backends for ARM and x86 platforms, with basic Windows build support. The recommended path today is to build from source and run GGUF models locally.

## TL;DR

```bash
git clone https://github.com/OpenBMB/ArcLight.git
cd ArcLight

cmake -B build -DARCLIGHT_BACKEND=AUTO -DNNML_USE_NUMA=OFF
cmake --build build --config Release -j 32

./build/al-gen \
    --model /path/to/MiniCPM5-1B-Q4_0.gguf \
    --prompt "Hello!" \
    --numa none --nodes 1 \
    --threads 4
```

## Preparing a Model

ArcLight uses the GGUF model format from [llama.cpp](https://github.com/ggml-org/llama.cpp). Download a GGUF checkpoint from Hugging Face or convert your own model by following the llama.cpp GGUF conversion workflow.

The current codebase includes model definitions for:

- MiniCPM5-1B
- Qwen3
- Llama2

For first-time testing, start with MiniCPM5-1B or another small GGUF model, preferably a quantized checkpoint such as `Q4_0`.

## Building

### Linux / x86 / ARM

```bash
git clone https://github.com/OpenBMB/ArcLight.git
cd ArcLight

cmake -B build -DARCLIGHT_BACKEND=AUTO
cmake --build build --config Release -j 32
```

`ARCLIGHT_BACKEND` can be set to:


| Value  | Meaning                                                           |
| ------ | ----------------------------------------------------------------- |
| `AUTO` | Select the backend from the target CPU architecture. Recommended. |
| `X86`  | Enable the x86 backend explicitly.                                |
| `NEON` | Enable the ARM NEON backend explicitly.                           |
| `NONE` | Build without architecture-specific backend code.                 |


Make sure your machine has a C++17-capable toolchain, such as GCC/G++ on Linux or MSVC on Windows.

### Windows

```bat
git clone https://github.com/OpenBMB/ArcLight.git
cd ArcLight

cmake -B build -G "Visual Studio 18 2026"
cmake --build build --config Release -j 32
```

On Windows, executables are emitted under the build output directory configured by CMake, typically `build\bin` for Visual Studio builds.

## Inference

ArcLight currently provides three command-line apps:

- `al-gen`: one-shot generation
- `al-chat`: interactive chat
- `al-ppl`: perplexity evaluation for one text

If you build from source, run them from the build directory, for example `./build/al-gen` on Linux.

### One-shot Generate

```bash
./build/al-gen \
    --model /path/to/MiniCPM5-1B-Q4_0.gguf \
    --prompt "Explain what unified memory means in one sentence." \
    --numa none --nodes 1 \
    --threads 4 \
    --max_length 4096 \
    --max_gen 256
```

For a Chinese prompt:

```bash
./build/al-gen \
    --model /path/to/MiniCPM5-1B-Q4_0.gguf \
    --prompt "用一句话解释什么是统一内存。" \
    --numa none --nodes 1 \
    --threads 4 \
    --max_gen 256
```

### Interactive Chat

```bash
./build/al-chat \
    --model /path/to/MiniCPM5-1B-Q4_0.gguf \
    --numa none --nodes 1 \
    --threads 4 \
    --max_length 4096 \
    --max_gen 512
```

Press `Ctrl+C` during generation to interrupt the current response. Press `Ctrl+C` while waiting for input to exit and print the performance profile.

### Perplexity

```bash
./build/al-ppl \
    --model /path/to/MiniCPM5-1B-Q4_0.gguf \
    --prompt "Good morning, Miss Lee!" \
    --numa none --nodes 1 \
    --threads 4
```

The app prints the evaluated text and a final `perplexity: ...` line.

## NUMA Modes

ArcLight supports single-node inference and cross-node tensor parallelism.


| Mode          | Required args             | When to use                                                                         |
| ------------- | ------------------------- | ----------------------------------------------------------------------------------- |
| `--numa none` | `--nodes 1`               | Single-node mode. Start here for correctness checks and small models.               |
| `--numa tp`   | `--nodes N` where `N > 1` | Cross-NUMA tensor parallelism. Use on many-core CPU machines for higher throughput. |
| `--numa pp`   | Not ready                 | Reserved for future pipeline parallelism; currently not implemented.                |


For tensor parallelism, `--nodes` should be a power of 2 in the current version. Choose `--threads` so that it can be evenly divided across NUMA nodes.

Example for a 4-node many-core machine:

```bash
./build/al-gen \
    --model /path/to/MiniCPM5-1B-Q4_0.gguf\
    --prompt "Hello!" \
    --numa tp --nodes 4 \
    --threads 32
```

## Recommended Settings


| Scenario                 | Suggested settings                                                         |
| ------------------------ | -------------------------------------------------------------------------- |
| First run / small model  | `--numa none --nodes 1 --threads <cores-on-one-node>`                      |
| Many-core CPU throughput | `--numa tp --nodes <power-of-2> --threads <total-threads>`                 |
| Longer context           | Increase `--max_length` and `--kv_gb`.                                     |
| Larger model             | Increase `--w_gb`, then tune `--a_gb` and `--work_gb` if allocation fails. |


## Q&A

### The program aborts immediately in single-node mode

Use `--numa none --nodes 1`. The current implementation requires `--nodes` to be exactly `1` when `--numa none` is selected.

### Tensor parallel mode fails to start

Use `--numa tp --nodes N` with `N > 1`. In the current version, `N` should be a power of 2. Also make sure `--threads` is large enough and divisible by `--nodes`.

### Pipeline parallelism does not work

`--numa pp` is planned but not implemented yet. Use `--numa none` or `--numa tp`.

### Model loading fails

Check that the model is a GGUF checkpoint from a supported model family. The current codebase includes Qwen3, Llama, and MiniCPM5 definitions. Also verify that `--w_gb` is large enough for the selected model.

### Inference runs out of memory

Increase `--a_gb`, `--kv_gb`, or `--work_gb`. Longer contexts require a larger KV cache, so `--max_length 8192` generally needs a larger `--kv_gb` than `--max_length 4096`.

## See Also

- [ArcLight README](https://github.com/OpenBMB/ArcLight/README.md)
- [llama.cpp GGUF documentation](https://github.com/ggml-org/llama.cpp)
- [MiniCPM5 MLX deployment guide](https://github.com/OpenBMB/MiniCPM/blob/minicpm5/docs/deployment/mlx.md)
